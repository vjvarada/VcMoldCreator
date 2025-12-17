/**
 * Fast C++ implementations for performance-critical mesh algorithms.
 * 
 * Provides optimized versions of:
 * - Dijkstra's algorithm for escape labeling
 * - Edge boundary label computation
 * 
 * Compiled with pybind11 for seamless Python integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ============================================================================
// MATH UTILITIES
// ============================================================================

using Vec3 = std::array<double, 3>;

inline Vec3 vec3_sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3 vec3_add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline Vec3 vec3_scale(const Vec3& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

inline double vec3_dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 vec3_cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline double vec3_length(const Vec3& a) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline Vec3 vec3_normalize(const Vec3& a) {
    double len = vec3_length(a);
    if (len < 1e-12) return {0, 0, 0};
    return vec3_scale(a, 1.0 / len);
}

// ============================================================================
// DIJKSTRA'S ALGORITHM FOR ESCAPE LABELING
// ============================================================================

/**
 * Result structure for Dijkstra escape labeling.
 */
struct DijkstraResult {
    py::array_t<int8_t> escape_labels;      // 1=H1, 2=H2, 0=unreachable
    py::array_t<int64_t> vertex_indices;    // Interior vertex indices
    py::array_t<double> distances;          // Shortest distances to boundary
    py::array_t<int64_t> destinations;      // Destination boundary vertices
    std::vector<std::vector<int64_t>> paths; // Escape paths
};

/**
 * Fast Dijkstra escape labeling from interior vertices to H1/H2 boundaries.
 * 
 * For each interior vertex (not on H1 or H2), finds the shortest weighted path
 * to either H1 or H2 boundary and labels the vertex accordingly.
 * 
 * @param edges (E, 2) array of edge vertex indices
 * @param edge_lengths (E,) array of edge lengths
 * @param edge_weights (E,) array of edge weights (higher = lower cost near part)
 * @param boundary_labels (N,) array: 0=interior, 1=H1, 2=H2, -1=inner boundary
 * @param use_weighted_edges Whether to use edge_length * edge_weight as cost
 * @return DijkstraResult with labels, distances, and paths
 */
DijkstraResult dijkstra_escape_labeling(
    py::array_t<int64_t> edges,
    py::array_t<double> edge_lengths,
    py::array_t<double> edge_weights,
    py::array_t<int8_t> boundary_labels,
    bool use_weighted_edges = true
) {
    // Get array info
    auto edges_buf = edges.unchecked<2>();
    auto lengths_buf = edge_lengths.unchecked<1>();
    auto weights_buf = edge_weights.unchecked<1>();
    auto labels_buf = boundary_labels.unchecked<1>();
    
    const int64_t n_edges = edges_buf.shape(0);
    const int64_t n_vertices = labels_buf.shape(0);
    
    // Compute edge costs
    std::vector<double> edge_costs(n_edges);
    for (int64_t i = 0; i < n_edges; i++) {
        if (use_weighted_edges) {
            edge_costs[i] = lengths_buf(i) * weights_buf(i);
        } else {
            edge_costs[i] = lengths_buf(i);
        }
    }
    
    // Build adjacency list: vertex -> [(neighbor, cost), ...]
    std::vector<std::vector<std::pair<int64_t, double>>> adjacency(n_vertices);
    for (int64_t i = 0; i < n_edges; i++) {
        int64_t v0 = edges_buf(i, 0);
        int64_t v1 = edges_buf(i, 1);
        double cost = edge_costs[i];
        adjacency[v0].emplace_back(v1, cost);
        adjacency[v1].emplace_back(v0, cost);
    }
    
    // Identify interior vertices and boundary sets
    std::vector<int64_t> interior_vertices;
    std::unordered_set<int64_t> h1_vertices, h2_vertices;
    
    for (int64_t i = 0; i < n_vertices; i++) {
        int8_t label = labels_buf(i);
        if (label == 1) {
            h1_vertices.insert(i);
        } else if (label == 2) {
            h2_vertices.insert(i);
        } else {
            // Interior vertices: label 0 (interior) or -1 (inner boundary/part surface)
            interior_vertices.push_back(i);
        }
    }
    
    const int64_t n_interior = interior_vertices.size();
    
    // Allocate result arrays
    auto result_labels = py::array_t<int8_t>(n_interior);
    auto result_distances = py::array_t<double>(n_interior);
    auto result_destinations = py::array_t<int64_t>(n_interior);
    auto result_indices = py::array_t<int64_t>(n_interior);
    
    auto labels_ptr = result_labels.mutable_unchecked<1>();
    auto dist_ptr = result_distances.mutable_unchecked<1>();
    auto dest_ptr = result_destinations.mutable_unchecked<1>();
    auto idx_ptr = result_indices.mutable_unchecked<1>();
    
    std::vector<std::vector<int64_t>> all_paths(n_interior);
    
    // Priority queue: (distance, vertex)
    using PQEntry = std::pair<double, int64_t>;
    
    // Run Dijkstra from each interior vertex
    for (int64_t idx = 0; idx < n_interior; idx++) {
        int64_t start_v = interior_vertices[idx];
        idx_ptr(idx) = start_v;
        
        // Initialize distances and predecessors
        std::vector<double> dist(n_vertices, std::numeric_limits<double>::infinity());
        std::vector<int64_t> predecessor(n_vertices, -1);
        std::vector<bool> visited(n_vertices, false);
        
        dist[start_v] = 0.0;
        
        // Min-heap priority queue
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
        pq.emplace(0.0, start_v);
        
        int64_t destination = -1;
        int8_t escape_label = 0;
        double final_dist = std::numeric_limits<double>::infinity();
        
        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            // Check if we reached a target boundary
            if (h1_vertices.count(u)) {
                escape_label = 1;
                final_dist = d;
                destination = u;
                break;
            } else if (h2_vertices.count(u)) {
                escape_label = 2;
                final_dist = d;
                destination = u;
                break;
            }
            
            // Propagate to neighbors
            for (const auto& [v, cost] : adjacency[u]) {
                if (visited[v]) continue;
                double new_dist = d + cost;
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    predecessor[v] = u;
                    pq.emplace(new_dist, v);
                }
            }
        }
        
        labels_ptr(idx) = escape_label;
        dist_ptr(idx) = final_dist;
        dest_ptr(idx) = destination;
        
        // Reconstruct path
        if (destination >= 0) {
            std::vector<int64_t> path;
            int64_t current = destination;
            while (current >= 0) {
                path.push_back(current);
                current = predecessor[current];
            }
            std::reverse(path.begin(), path.end());
            all_paths[idx] = std::move(path);
        }
    }
    
    DijkstraResult result;
    result.escape_labels = result_labels;
    result.vertex_indices = result_indices;
    result.distances = result_distances;
    result.destinations = result_destinations;
    result.paths = std::move(all_paths);
    
    return result;
}


// ============================================================================
// EDGE BOUNDARY LABEL COMPUTATION
// ============================================================================

/**
 * Compute edge boundary labels based on vertex labels and boundary surface membership.
 * 
 * Edge labels:
 *   0  = Interior edge (not on boundary surface)
 *   1  = H1 boundary edge (both vertices on H1)
 *   2  = H2 boundary edge (both vertices on H2)
 *  -1  = Inner boundary edge (both vertices on inner/part boundary)
 *  -2  = Mixed boundary edge (different boundary types)
 * 
 * @param edges (E, 2) array of edge vertex indices (sorted)
 * @param boundary_labels (N,) array: 0=interior, 1=H1, 2=H2, -1=inner
 * @param boundary_faces (F, 3) array of boundary mesh face vertex indices
 * @param boundary_mesh_vertices (B, 3) array of boundary mesh vertex positions
 * @param tet_vertices (N, 3) array of tetrahedral mesh vertex positions
 * @return (E,) int8 array of edge boundary labels
 */
py::array_t<int8_t> compute_edge_boundary_labels(
    py::array_t<int64_t> edges,
    py::array_t<int8_t> boundary_labels,
    py::array_t<int64_t> boundary_faces,
    py::array_t<double> boundary_mesh_vertices,
    py::array_t<double> tet_vertices
) {
    auto edges_buf = edges.unchecked<2>();
    auto labels_buf = boundary_labels.unchecked<1>();
    auto faces_buf = boundary_faces.unchecked<2>();
    auto bmv_buf = boundary_mesh_vertices.unchecked<2>();
    auto tv_buf = tet_vertices.unchecked<2>();
    
    const int64_t n_edges = edges_buf.shape(0);
    const int64_t n_boundary_verts = bmv_buf.shape(0);
    const int64_t n_tet_verts = tv_buf.shape(0);
    const int64_t n_faces = faces_buf.shape(0);
    
    // Build mapping from boundary mesh vertices to tet vertices (nearest neighbor)
    std::vector<int64_t> boundary_to_tet(n_boundary_verts);
    
    for (int64_t bi = 0; bi < n_boundary_verts; bi++) {
        double bx = bmv_buf(bi, 0);
        double by = bmv_buf(bi, 1);
        double bz = bmv_buf(bi, 2);
        
        double min_dist_sq = std::numeric_limits<double>::infinity();
        int64_t nearest = 0;
        
        for (int64_t ti = 0; ti < n_tet_verts; ti++) {
            double dx = tv_buf(ti, 0) - bx;
            double dy = tv_buf(ti, 1) - by;
            double dz = tv_buf(ti, 2) - bz;
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest = ti;
            }
        }
        boundary_to_tet[bi] = nearest;
    }
    
    // Build set of boundary surface edges (in tet vertex indices)
    // Using a custom hash for pairs
    struct PairHash {
        size_t operator()(const std::pair<int64_t, int64_t>& p) const {
            return std::hash<int64_t>()(p.first) ^ (std::hash<int64_t>()(p.second) << 1);
        }
    };
    
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> boundary_surface_edges;
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        int64_t v0_tet = boundary_to_tet[faces_buf(fi, 0)];
        int64_t v1_tet = boundary_to_tet[faces_buf(fi, 1)];
        int64_t v2_tet = boundary_to_tet[faces_buf(fi, 2)];
        
        // Add edges (sorted)
        auto make_edge = [](int64_t a, int64_t b) {
            return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        };
        
        boundary_surface_edges.insert(make_edge(v0_tet, v1_tet));
        boundary_surface_edges.insert(make_edge(v1_tet, v2_tet));
        boundary_surface_edges.insert(make_edge(v2_tet, v0_tet));
    }
    
    // Compute edge labels
    auto result = py::array_t<int8_t>(n_edges);
    auto result_ptr = result.mutable_unchecked<1>();
    
    for (int64_t ei = 0; ei < n_edges; ei++) {
        int64_t v0 = edges_buf(ei, 0);
        int64_t v1 = edges_buf(ei, 1);
        
        // Check if edge is on boundary surface
        auto edge_key = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
        bool on_surface = boundary_surface_edges.count(edge_key) > 0;
        
        if (!on_surface) {
            result_ptr(ei) = 0;  // Interior edge
            continue;
        }
        
        // Get vertex labels
        int8_t label0 = labels_buf(v0);
        int8_t label1 = labels_buf(v1);
        
        // Classify based on labels
        if (label0 == 1 && label1 == 1) {
            result_ptr(ei) = 1;  // H1 edge
        } else if (label0 == 2 && label1 == 2) {
            result_ptr(ei) = 2;  // H2 edge
        } else if (label0 == -1 && label1 == -1) {
            result_ptr(ei) = -1;  // Inner boundary edge
        } else {
            result_ptr(ei) = -2;  // Mixed boundary edge
        }
    }
    
    return result;
}


// ============================================================================
// OPTIMIZED EDGE-TO-TRIANGLE MAPPING FOR FAST QUERIES
// ============================================================================

/**
 * Build a fast edge-to-triangle lookup for boundary mesh.
 * Returns edge indices for quick spatial queries.
 */
py::array_t<int64_t> build_boundary_to_tet_mapping(
    py::array_t<double> boundary_vertices,
    py::array_t<double> tet_vertices
) {
    auto bv_buf = boundary_vertices.unchecked<2>();
    auto tv_buf = tet_vertices.unchecked<2>();
    
    const int64_t n_boundary = bv_buf.shape(0);
    const int64_t n_tet = tv_buf.shape(0);
    
    auto result = py::array_t<int64_t>(n_boundary);
    auto result_ptr = result.mutable_unchecked<1>();
    
    // For each boundary vertex, find nearest tet vertex
    for (int64_t bi = 0; bi < n_boundary; bi++) {
        double bx = bv_buf(bi, 0);
        double by = bv_buf(bi, 1);
        double bz = bv_buf(bi, 2);
        
        double min_dist_sq = std::numeric_limits<double>::infinity();
        int64_t nearest = 0;
        
        for (int64_t ti = 0; ti < n_tet; ti++) {
            double dx = tv_buf(ti, 0) - bx;
            double dy = tv_buf(ti, 1) - by;
            double dz = tv_buf(ti, 2) - bz;
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest = ti;
            }
        }
        result_ptr(bi) = nearest;
    }
    
    return result;
}


// ============================================================================
// SECONDARY CUTS - TRIANGLE INTERSECTION ALGORITHMS
// ============================================================================

/**
 * Möller–Trumbore ray-triangle intersection algorithm.
 * 
 * Tests if a line segment (p0, p1) intersects a triangle (v0, v1, v2).
 * 
 * @param p0, p1 Segment endpoints
 * @param v0, v1, v2 Triangle vertices
 * @param sensitivity Controls tolerance (0=strict, 1=lenient)
 * @return true if segment intersects triangle
 */
inline bool segment_intersects_triangle(
    const Vec3& p0, const Vec3& p1,
    const Vec3& v0, const Vec3& v1, const Vec3& v2
) {
    constexpr double EPSILON = 1e-9;
    
    Vec3 direction = vec3_sub(p1, p0);
    double seg_length = vec3_length(direction);
    if (seg_length < EPSILON) return false;
    
    direction = vec3_scale(direction, 1.0 / seg_length);  // Normalize
    
    // Triangle edges
    Vec3 edge1 = vec3_sub(v1, v0);
    Vec3 edge2 = vec3_sub(v2, v0);
    
    // Begin calculating determinant
    Vec3 h = vec3_cross(direction, edge2);
    double a = vec3_dot(edge1, h);
    
    if (std::abs(a) < EPSILON) {
        // Ray is parallel to triangle
        return false;
    }
    
    double f = 1.0 / a;
    Vec3 s = vec3_sub(p0, v0);
    double u = f * vec3_dot(s, h);
    
    if (u < -EPSILON || u > 1.0 + EPSILON) {
        return false;
    }
    
    Vec3 q = vec3_cross(s, edge1);
    double v = f * vec3_dot(direction, q);
    
    if (v < -EPSILON || u + v > 1.0 + EPSILON) {
        return false;
    }
    
    // Compute t to find intersection point
    double t = f * vec3_dot(edge2, q);
    
    // Check if intersection is within the segment (boolean - no tolerance scaling)
    return (t >= -EPSILON) && (t <= seg_length + EPSILON);
}

/**
 * Check if two triangles intersect.
 * 
 * Two triangles intersect if any edge of one passes through the other.
 */
inline bool triangles_intersect(
    const Vec3& t1_v0, const Vec3& t1_v1, const Vec3& t1_v2,
    const Vec3& t2_v0, const Vec3& t2_v1, const Vec3& t2_v2
) {
    // Check edges of tri1 against tri2
    if (segment_intersects_triangle(t1_v0, t1_v1, t2_v0, t2_v1, t2_v2)) return true;
    if (segment_intersects_triangle(t1_v1, t1_v2, t2_v0, t2_v1, t2_v2)) return true;
    if (segment_intersects_triangle(t1_v2, t1_v0, t2_v0, t2_v1, t2_v2)) return true;
    
    // Check edges of tri2 against tri1
    if (segment_intersects_triangle(t2_v0, t2_v1, t1_v0, t1_v1, t1_v2)) return true;
    if (segment_intersects_triangle(t2_v1, t2_v2, t1_v0, t1_v1, t1_v2)) return true;
    if (segment_intersects_triangle(t2_v2, t2_v0, t1_v0, t1_v1, t1_v2)) return true;
    
    return false;
}

/**
 * Result structure for secondary cuts detection.
 */
struct SecondaryCutsResult {
    std::vector<std::pair<int64_t, int64_t>> cutting_edges;  // List of (vi, vj) pairs
    int64_t n_membranes_checked;
    int64_t n_intersections_found;
};

/**
 * Build membrane triangles from escape paths.
 * 
 * Returns a vector of triangles (each triangle is 3 Vec3 vertices).
 */
std::vector<std::array<Vec3, 3>> build_membrane_triangles(
    const std::vector<int64_t>& path_i,
    const std::vector<int64_t>& path_j,
    const std::vector<int64_t>& boundary_path,
    const double* tet_vertices,  // Nx3 flattened
    const double* boundary_vertices,  // Bx3 flattened
    double min_thickness
) {
    std::vector<std::array<Vec3, 3>> triangles;
    
    size_t n_i = path_i.size();
    size_t n_j = path_j.size();
    
    if (n_i < 1 || n_j < 1) return triangles;
    
    // Get positions along path_i
    std::vector<Vec3> path_i_positions(n_i);
    for (size_t k = 0; k < n_i; k++) {
        int64_t idx = path_i[k];
        path_i_positions[k] = {
            tet_vertices[idx * 3 + 0],
            tet_vertices[idx * 3 + 1],
            tet_vertices[idx * 3 + 2]
        };
    }
    
    // Get positions along path_j
    std::vector<Vec3> path_j_positions(n_j);
    for (size_t k = 0; k < n_j; k++) {
        int64_t idx = path_j[k];
        path_j_positions[k] = {
            tet_vertices[idx * 3 + 0],
            tet_vertices[idx * 3 + 1],
            tet_vertices[idx * 3 + 2]
        };
    }
    
    // Check if paths are too similar (flat surface case)
    if (min_thickness > 0) {
        double max_separation = 0.0;
        size_t n_check = std::min({n_i, n_j, size_t(5)});
        for (size_t k = 0; k < n_check; k++) {
            double t = (n_check > 1) ? double(k) / (n_check - 1) : 0.0;
            size_t idx_i = std::min(size_t(t * (n_i - 1)), n_i - 1);
            size_t idx_j = std::min(size_t(t * (n_j - 1)), n_j - 1);
            double sep = vec3_length(vec3_sub(path_i_positions[idx_i], path_j_positions[idx_j]));
            max_separation = std::max(max_separation, sep);
        }
        if (max_separation < min_thickness) {
            return triangles;  // Membrane too thin
        }
    }
    
    // Resample paths to have same number of points
    size_t n_samples = std::max({n_i, n_j, size_t(5)});
    
    std::vector<Vec3> path_i_resampled(n_samples);
    std::vector<Vec3> path_j_resampled(n_samples);
    
    for (size_t s = 0; s < n_samples; s++) {
        double t = (n_samples > 1) ? double(s) / (n_samples - 1) : 0.0;
        
        // Interpolate along path_i
        size_t idx_i = std::min(size_t(t * (n_i - 1)), n_i - 1);
        double frac_i = t * (n_i - 1) - idx_i;
        if (idx_i < n_i - 1) {
            path_i_resampled[s] = vec3_add(
                vec3_scale(path_i_positions[idx_i], 1.0 - frac_i),
                vec3_scale(path_i_positions[idx_i + 1], frac_i)
            );
        } else {
            path_i_resampled[s] = path_i_positions[idx_i];
        }
        
        // Interpolate along path_j
        size_t idx_j = std::min(size_t(t * (n_j - 1)), n_j - 1);
        double frac_j = t * (n_j - 1) - idx_j;
        if (idx_j < n_j - 1) {
            path_j_resampled[s] = vec3_add(
                vec3_scale(path_j_positions[idx_j], 1.0 - frac_j),
                vec3_scale(path_j_positions[idx_j + 1], frac_j)
            );
        } else {
            path_j_resampled[s] = path_j_positions[idx_j];
        }
    }
    
    // Create triangles between the two paths
    for (size_t s = 0; s < n_samples - 1; s++) {
        const Vec3& p0 = path_i_resampled[s];
        const Vec3& p1 = path_i_resampled[s + 1];
        const Vec3& p2 = path_j_resampled[s];
        const Vec3& p3 = path_j_resampled[s + 1];
        
        triangles.push_back({p0, p1, p2});
        triangles.push_back({p1, p3, p2});
    }
    
    // Add boundary path lid
    if (boundary_path.size() > 1) {
        const Vec3& wi_pos = path_i_resampled.back();
        const Vec3& wj_pos = path_j_resampled.back();
        Vec3 mid = vec3_scale(vec3_add(wi_pos, wj_pos), 0.5);
        
        for (size_t i = 0; i < boundary_path.size() - 1; i++) {
            int64_t bp0_idx = boundary_path[i];
            int64_t bp1_idx = boundary_path[i + 1];
            Vec3 bp0 = {
                boundary_vertices[bp0_idx * 3 + 0],
                boundary_vertices[bp0_idx * 3 + 1],
                boundary_vertices[bp0_idx * 3 + 2]
            };
            Vec3 bp1 = {
                boundary_vertices[bp1_idx * 3 + 0],
                boundary_vertices[bp1_idx * 3 + 1],
                boundary_vertices[bp1_idx * 3 + 2]
            };
            triangles.push_back({bp0, bp1, mid});
        }
    }
    
    return triangles;
}

/**
 * Find secondary cutting edges using C++ with OpenMP parallelization.
 * 
 * This checks if the membrane between same-label interior vertices
 * intersects the part mesh (represented by seed triangles).
 * 
 * Uses boolean intersection detection - no tolerance scaling.
 * Requires minimum number of intersections to trigger (reduces false positives).
 * 
 * @param membrane_edge_vi Array of vi vertex indices for each membrane
 * @param membrane_edge_vj Array of vj vertex indices for each membrane
 * @param membrane_path_i Flattened array of path_i vertex indices
 * @param membrane_path_i_offsets Offset into path_i array for each membrane
 * @param membrane_path_j Flattened array of path_j vertex indices
 * @param membrane_path_j_offsets Offset into path_j array for each membrane
 * @param membrane_boundary_path Flattened array of boundary path indices
 * @param membrane_boundary_offsets Offset into boundary_path array for each membrane
 * @param seed_triangles (T, 3, 3) array of seed triangle positions
 * @param seed_triangle_vertices (T, 3) array of seed triangle vertex indices
 * @param tet_vertices (N, 3) array of tetrahedral mesh vertices
 * @param boundary_vertices (B, 3) array of boundary mesh vertices
 * @param min_intersection_count Minimum number of segment-triangle intersections required (1-50)
 * @param min_membrane_thickness Skip membranes thinner than this
 * @return SecondaryCutsResult with cutting edges
 */
SecondaryCutsResult find_secondary_cuts_cpp(
    py::array_t<int64_t> membrane_edge_vi,
    py::array_t<int64_t> membrane_edge_vj,
    py::array_t<int64_t> membrane_path_i,
    py::array_t<int64_t> membrane_path_i_offsets,
    py::array_t<int64_t> membrane_path_j,
    py::array_t<int64_t> membrane_path_j_offsets,
    py::array_t<int64_t> membrane_boundary_path,
    py::array_t<int64_t> membrane_boundary_offsets,
    py::array_t<double> seed_triangles,
    py::array_t<int64_t> seed_triangle_vertices,
    py::array_t<double> tet_vertices,
    py::array_t<double> boundary_vertices,
    int64_t min_intersection_count,
    double min_membrane_thickness
) {
    // Get array buffers
    auto vi_buf = membrane_edge_vi.unchecked<1>();
    auto vj_buf = membrane_edge_vj.unchecked<1>();
    auto path_i_buf = membrane_path_i.unchecked<1>();
    auto path_i_off_buf = membrane_path_i_offsets.unchecked<1>();
    auto path_j_buf = membrane_path_j.unchecked<1>();
    auto path_j_off_buf = membrane_path_j_offsets.unchecked<1>();
    auto boundary_buf = membrane_boundary_path.unchecked<1>();
    auto boundary_off_buf = membrane_boundary_offsets.unchecked<1>();
    auto seed_tri_buf = seed_triangles.unchecked<3>();  // (T, 3, 3)
    auto seed_vert_buf = seed_triangle_vertices.unchecked<2>();  // (T, 3)
    auto tet_vert_buf = tet_vertices.unchecked<2>();  // (N, 3)
    auto bound_vert_buf = boundary_vertices.unchecked<2>();  // (B, 3)
    
    const int64_t n_membranes = vi_buf.shape(0);
    const int64_t n_seed_triangles = seed_tri_buf.shape(0);
    
    // Pre-load seed triangles into local arrays for faster access
    std::vector<std::array<Vec3, 3>> seed_tris(n_seed_triangles);
    std::vector<std::array<int64_t, 3>> seed_verts(n_seed_triangles);
    
    for (int64_t t = 0; t < n_seed_triangles; t++) {
        seed_tris[t][0] = {seed_tri_buf(t, 0, 0), seed_tri_buf(t, 0, 1), seed_tri_buf(t, 0, 2)};
        seed_tris[t][1] = {seed_tri_buf(t, 1, 0), seed_tri_buf(t, 1, 1), seed_tri_buf(t, 1, 2)};
        seed_tris[t][2] = {seed_tri_buf(t, 2, 0), seed_tri_buf(t, 2, 1), seed_tri_buf(t, 2, 2)};
        seed_verts[t] = {seed_vert_buf(t, 0), seed_vert_buf(t, 1), seed_vert_buf(t, 2)};
    }
    
    // Get raw pointers for faster access
    const double* tet_verts_ptr = tet_vert_buf.data(0, 0);
    const double* bound_verts_ptr = bound_vert_buf.data(0, 0);
    
    // Thread-local results (to avoid locking)
    std::vector<std::vector<std::pair<int64_t, int64_t>>> thread_results;
    int max_threads = 1;
    
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    thread_results.resize(max_threads);
    
    std::vector<int64_t> thread_intersections(max_threads, 0);
    
    // Process membranes in parallel
#pragma omp parallel for schedule(dynamic, 10)
    for (int64_t m = 0; m < n_membranes; m++) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        
        int64_t vi = vi_buf(m);
        int64_t vj = vj_buf(m);
        
        // Extract path_i for this membrane
        int64_t pi_start = path_i_off_buf(m);
        int64_t pi_end = (m + 1 < n_membranes) ? path_i_off_buf(m + 1) : membrane_path_i.size();
        std::vector<int64_t> path_i;
        for (int64_t i = pi_start; i < pi_end; i++) {
            path_i.push_back(path_i_buf(i));
        }
        
        // Extract path_j for this membrane
        int64_t pj_start = path_j_off_buf(m);
        int64_t pj_end = (m + 1 < n_membranes) ? path_j_off_buf(m + 1) : membrane_path_j.size();
        std::vector<int64_t> path_j;
        for (int64_t i = pj_start; i < pj_end; i++) {
            path_j.push_back(path_j_buf(i));
        }
        
        // Extract boundary_path for this membrane
        int64_t bp_start = boundary_off_buf(m);
        int64_t bp_end = (m + 1 < n_membranes) ? boundary_off_buf(m + 1) : membrane_boundary_path.size();
        std::vector<int64_t> boundary_path;
        for (int64_t i = bp_start; i < bp_end; i++) {
            boundary_path.push_back(boundary_buf(i));
        }
        
        // Build membrane triangles
        auto membrane_triangles = build_membrane_triangles(
            path_i, path_j, boundary_path,
            tet_verts_ptr, bound_verts_ptr,
            min_membrane_thickness
        );
        
        if (membrane_triangles.empty()) continue;
        
        // Build skip vertices set (vertices adjacent to edge)
        std::unordered_set<int64_t> skip_vertices;
        skip_vertices.insert(vi);
        skip_vertices.insert(vj);
        size_t n_skip = std::min(size_t(3), path_i.size());
        for (size_t k = 0; k < n_skip; k++) skip_vertices.insert(path_i[k]);
        n_skip = std::min(size_t(3), path_j.size());
        for (size_t k = 0; k < n_skip; k++) skip_vertices.insert(path_j[k]);
        
        // Count intersections with seed triangles
        int64_t intersection_count = 0;
        
        for (int64_t t = 0; t < n_seed_triangles; t++) {
            // Skip triangles sharing vertices with membrane edge
            const auto& sv = seed_verts[t];
            if (skip_vertices.count(sv[0]) || skip_vertices.count(sv[1]) || skip_vertices.count(sv[2])) {
                continue;
            }
            
            const auto& seed_tri = seed_tris[t];
            
            // Check each membrane triangle against this seed triangle
            for (const auto& mem_tri : membrane_triangles) {
                if (triangles_intersect(
                    mem_tri[0], mem_tri[1], mem_tri[2],
                    seed_tri[0], seed_tri[1], seed_tri[2]
                )) {
                    intersection_count++;
                    // Early exit if we've reached the threshold
                    if (intersection_count >= min_intersection_count) {
                        break;
                    }
                }
            }
            if (intersection_count >= min_intersection_count) {
                break;
            }
        }
        
        if (intersection_count >= min_intersection_count) {
            thread_results[thread_id].emplace_back(vi, vj);
            thread_intersections[thread_id]++;
        }
    }
    
    // Merge thread results
    SecondaryCutsResult result;
    result.n_membranes_checked = n_membranes;
    result.n_intersections_found = 0;
    
    for (int t = 0; t < max_threads; t++) {
        for (const auto& edge : thread_results[t]) {
            result.cutting_edges.push_back(edge);
        }
        result.n_intersections_found += thread_intersections[t];
    }
    
    return result;
}


// ============================================================================
// MOLD HALF CLASSIFICATION WITH OPENMP
// ============================================================================

/**
 * Result structure for mold half classification.
 */
struct MoldHalfClassificationResult {
    py::array_t<int8_t> side_labels;        // 1=H1, 2=H2 for each outer triangle
    py::array_t<int64_t> outer_indices;     // Indices of outer triangles
    int64_t n_h1;
    int64_t n_h2;
};

/**
 * Build triangle adjacency from face array using edge-based matching.
 * Returns CSR-format adjacency: adj_indices, adj_ptr
 * 
 * Much faster than Python loop-based construction.
 */
std::pair<std::vector<int64_t>, std::vector<int64_t>> build_triangle_adjacency_cpp(
    py::array_t<int64_t> faces
) {
    auto faces_buf = faces.unchecked<2>();
    const int64_t n_faces = faces_buf.shape(0);
    
    // Build edge -> face list mapping
    // Use sorted vertex pairs as edge keys
    struct PairHash {
        size_t operator()(const std::pair<int64_t, int64_t>& p) const {
            return std::hash<int64_t>()(p.first) ^ (std::hash<int64_t>()(p.second) << 1);
        }
    };
    
    std::unordered_map<std::pair<int64_t, int64_t>, std::vector<int64_t>, PairHash> edge_to_faces;
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        int64_t v0 = faces_buf(fi, 0);
        int64_t v1 = faces_buf(fi, 1);
        int64_t v2 = faces_buf(fi, 2);
        
        auto make_edge = [](int64_t a, int64_t b) {
            return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        };
        
        edge_to_faces[make_edge(v0, v1)].push_back(fi);
        edge_to_faces[make_edge(v1, v2)].push_back(fi);
        edge_to_faces[make_edge(v2, v0)].push_back(fi);
    }
    
    // Build adjacency lists
    std::vector<std::vector<int64_t>> adjacency(n_faces);
    
    for (const auto& [edge, face_list] : edge_to_faces) {
        // Each edge can have at most 2 faces (manifold mesh)
        if (face_list.size() == 2) {
            adjacency[face_list[0]].push_back(face_list[1]);
            adjacency[face_list[1]].push_back(face_list[0]);
        }
    }
    
    // Convert to CSR format
    std::vector<int64_t> adj_indices;
    std::vector<int64_t> adj_ptr;
    adj_ptr.push_back(0);
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        for (int64_t neighbor : adjacency[fi]) {
            adj_indices.push_back(neighbor);
        }
        adj_ptr.push_back(adj_indices.size());
    }
    
    return {adj_indices, adj_ptr};
}

/**
 * Fast mold half classification using greedy region growing with OpenMP.
 * Takes faces array directly and builds adjacency internally (faster than Python).
 * 
 * @param face_normals (F, 3) array of face normals
 * @param faces (F, 3) array of face vertex indices
 * @param outer_indices (O,) array of outer triangle indices to classify
 * @param d1, d2 Parting directions (normalized)
 * @return MoldHalfClassificationResult with labels
 */
MoldHalfClassificationResult classify_mold_halves_from_faces_cpp(
    py::array_t<double> face_normals,
    py::array_t<int64_t> faces,
    py::array_t<int64_t> outer_indices,
    py::array_t<double> d1_arr,
    py::array_t<double> d2_arr
) {
    auto normals_buf = face_normals.unchecked<2>();
    auto faces_buf = faces.unchecked<2>();
    auto outer_buf = outer_indices.unchecked<1>();
    auto d1_buf = d1_arr.unchecked<1>();
    auto d2_buf = d2_arr.unchecked<1>();
    
    const int64_t n_faces = faces_buf.shape(0);
    const int64_t n_outer = outer_buf.shape(0);
    
    if (n_outer == 0) {
        MoldHalfClassificationResult result;
        result.side_labels = py::array_t<int8_t>(0);
        result.outer_indices = py::array_t<int64_t>(0);
        result.n_h1 = 0;
        result.n_h2 = 0;
        return result;
    }
    
    // Build adjacency from faces using edge-based matching (parallel-friendly)
    struct PairHash {
        size_t operator()(const std::pair<int64_t, int64_t>& p) const {
            return std::hash<int64_t>()(p.first) ^ (std::hash<int64_t>()(p.second) << 1);
        }
    };
    
    std::unordered_map<std::pair<int64_t, int64_t>, std::vector<int64_t>, PairHash> edge_to_faces;
    edge_to_faces.reserve(n_faces * 3);
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        int64_t v0 = faces_buf(fi, 0);
        int64_t v1 = faces_buf(fi, 1);
        int64_t v2 = faces_buf(fi, 2);
        
        auto make_edge = [](int64_t a, int64_t b) {
            return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        };
        
        edge_to_faces[make_edge(v0, v1)].push_back(fi);
        edge_to_faces[make_edge(v1, v2)].push_back(fi);
        edge_to_faces[make_edge(v2, v0)].push_back(fi);
    }
    
    // Build CSR adjacency
    std::vector<std::vector<int64_t>> adjacency(n_faces);
    for (const auto& [edge, face_list] : edge_to_faces) {
        if (face_list.size() == 2) {
            adjacency[face_list[0]].push_back(face_list[1]);
            adjacency[face_list[1]].push_back(face_list[0]);
        }
    }
    
    std::vector<int64_t> adj_indices;
    std::vector<int64_t> adj_ptr;
    adj_ptr.reserve(n_faces + 1);
    adj_ptr.push_back(0);
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        for (int64_t neighbor : adjacency[fi]) {
            adj_indices.push_back(neighbor);
        }
        adj_ptr.push_back(adj_indices.size());
    }
    
    // Extract directions
    Vec3 d1 = {d1_buf(0), d1_buf(1), d1_buf(2)};
    Vec3 d2 = {d2_buf(0), d2_buf(1), d2_buf(2)};
    
    // Create mapping from global face index to local index
    std::vector<int64_t> global_to_local(n_faces, -1);
    for (int64_t i = 0; i < n_outer; i++) {
        global_to_local[outer_buf(i)] = i;
    }
    
    // Compute dot products with parting directions (parallel)
    std::vector<double> dot1(n_outer), dot2(n_outer);
    
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t fi = outer_buf(i);
        Vec3 normal = {normals_buf(fi, 0), normals_buf(fi, 1), normals_buf(fi, 2)};
        dot1[i] = vec3_dot(normal, d1);
        dot2[i] = vec3_dot(normal, d2);
    }
    
    // Find seed faces
    int64_t f1_local = 0, f2_local = 0;
    double best1 = -std::numeric_limits<double>::infinity();
    double best2 = -std::numeric_limits<double>::infinity();
    
    for (int64_t i = 0; i < n_outer; i++) {
        if (dot1[i] > best1) { best1 = dot1[i]; f1_local = i; }
        if (dot2[i] > best2) { best2 = dot2[i]; f2_local = i; }
    }
    
    // Initialize labels: -1 = unassigned, 1 = H1, 2 = H2
    std::vector<int8_t> labels(n_outer, -1);
    labels[f1_local] = 1;
    labels[f2_local] = 2;
    
    // Priority queues for greedy expansion
    using PQEntry = std::pair<double, int64_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq1, pq2;
    std::vector<bool> in_queue(n_outer, false);
    
    auto add_neighbors_to_queue = [&](int64_t local_idx, 
                                       std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>>& pq,
                                       const std::vector<double>& dots) {
        int64_t global_idx = outer_buf(local_idx);
        int64_t start = adj_ptr[global_idx];
        int64_t end = adj_ptr[global_idx + 1];
        
        for (int64_t j = start; j < end; j++) {
            int64_t neighbor_global = adj_indices[j];
            int64_t neighbor_local = global_to_local[neighbor_global];
            
            if (neighbor_local < 0) continue;
            if (labels[neighbor_local] >= 0) continue;
            if (in_queue[neighbor_local]) continue;
            
            pq.emplace(-dots[neighbor_local], neighbor_local);
            in_queue[neighbor_local] = true;
        }
    };
    
    add_neighbors_to_queue(f1_local, pq1, dot1);
    add_neighbors_to_queue(f2_local, pq2, dot2);
    
    // Greedy expansion
    while (!pq1.empty() || !pq2.empty()) {
        while (!pq1.empty()) {
            auto [neg_score, local_idx] = pq1.top();
            pq1.pop();
            if (labels[local_idx] >= 0) continue;
            
            if (dot1[local_idx] >= dot2[local_idx]) {
                labels[local_idx] = 1;
                add_neighbors_to_queue(local_idx, pq1, dot1);
                break;
            } else {
                pq2.emplace(-dot2[local_idx], local_idx);
            }
        }
        
        while (!pq2.empty()) {
            auto [neg_score, local_idx] = pq2.top();
            pq2.pop();
            if (labels[local_idx] >= 0) continue;
            
            if (dot2[local_idx] >= dot1[local_idx]) {
                labels[local_idx] = 2;
                add_neighbors_to_queue(local_idx, pq2, dot2);
                break;
            } else {
                pq1.emplace(-dot1[local_idx], local_idx);
            }
        }
    }
    
    // Handle unassigned
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        if (labels[i] < 0) {
            labels[i] = (dot1[i] >= dot2[i]) ? 1 : 2;
        }
    }
    
    // Morphological smoothing (simplified - fewer iterations for speed)
    std::vector<std::vector<int64_t>> local_adj(n_outer);
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t global_idx = outer_buf(i);
        int64_t start = adj_ptr[global_idx];
        int64_t end = adj_ptr[global_idx + 1];
        for (int64_t j = start; j < end; j++) {
            int64_t neighbor_local = global_to_local[adj_indices[j]];
            if (neighbor_local >= 0) {
                local_adj[i].push_back(neighbor_local);
            }
        }
    }
    
    std::vector<int32_t> neighbor_counts(n_outer);
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        neighbor_counts[i] = static_cast<int32_t>(local_adj[i].size());
    }
    
    // Laplacian smoothing only (skip morphological for speed)
    for (int smooth_iter = 0; smooth_iter < 5; smooth_iter++) {
        std::vector<int32_t> h1_counts(n_outer, 0);
        std::vector<int32_t> h2_counts(n_outer, 0);
        
#pragma omp parallel for
        for (int64_t i = 0; i < n_outer; i++) {
            for (int64_t neighbor : local_adj[i]) {
                if (labels[neighbor] == 1) h1_counts[i]++;
                else if (labels[neighbor] == 2) h2_counts[i]++;
            }
        }
        
        std::vector<bool> flip_to_h1(n_outer, false);
        std::vector<bool> flip_to_h2(n_outer, false);
        
#pragma omp parallel for
        for (int64_t i = 0; i < n_outer; i++) {
            if (neighbor_counts[i] == 0) continue;
            if (labels[i] == 1 && h2_counts[i] > h1_counts[i] * 3 / 2) flip_to_h2[i] = true;
            else if (labels[i] == 2 && h1_counts[i] > h2_counts[i] * 3 / 2) flip_to_h1[i] = true;
        }
        
        bool any_flip = false;
        for (int64_t i = 0; i < n_outer; i++) {
            if (flip_to_h1[i]) { labels[i] = 1; any_flip = true; }
            if (flip_to_h2[i]) { labels[i] = 2; any_flip = true; }
        }
        if (!any_flip) break;
    }
    
    // Re-apply strong directional constraints
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        if (dot1[i] > 0.7 && dot1[i] > dot2[i] + 0.3) labels[i] = 1;
        else if (dot2[i] > 0.7 && dot2[i] > dot1[i] + 0.3) labels[i] = 2;
    }
    
    // Count results
    int64_t n_h1 = 0, n_h2 = 0;
    for (int64_t i = 0; i < n_outer; i++) {
        if (labels[i] == 1) n_h1++;
        else if (labels[i] == 2) n_h2++;
    }
    
    // Build result
    auto result_labels = py::array_t<int8_t>(n_outer);
    auto result_labels_ptr = result_labels.mutable_unchecked<1>();
    for (int64_t i = 0; i < n_outer; i++) {
        result_labels_ptr(i) = labels[i];
    }
    
    MoldHalfClassificationResult result;
    result.side_labels = result_labels;
    result.outer_indices = outer_indices;
    result.n_h1 = n_h1;
    result.n_h2 = n_h2;
    
    return result;
}

/**
 * Complete mold half classification result with all computed data.
 */
struct CompleteMoldHalfResult {
    py::array_t<int8_t> face_labels;     // (F,) -1=inner, 0=boundary_zone, 1=H1, 2=H2
    py::array_t<uint8_t> face_colors;    // (F, 4) RGBA colors
    int64_t n_h1;
    int64_t n_h2;
    int64_t n_boundary_zone;
    int64_t n_inner;
};

/**
 * Complete mold half classification pipeline in C++.
 * Handles: outer boundary detection, classification, boundary zone, and coloring.
 * 
 * @param face_normals (F, 3) face normals
 * @param faces (F, 3) face vertex indices  
 * @param mesh_vertices (V, 3) mesh vertices
 * @param tet_vertices (T, 3) tet mesh vertices (for label lookup)
 * @param tet_boundary_labels (T,) boundary labels: -1=inner, 1=H1, 2=H2, 0=interior
 * @param d1, d2 parting directions
 * @param boundary_zone_hops BFS expansion hops for boundary zone
 */
CompleteMoldHalfResult classify_mold_halves_complete_cpp(
    py::array_t<double> face_normals,
    py::array_t<int64_t> faces,
    py::array_t<double> mesh_vertices,
    py::array_t<double> tet_vertices,
    py::array_t<int8_t> tet_boundary_labels,
    py::array_t<double> d1_arr,
    py::array_t<double> d2_arr,
    int64_t boundary_zone_hops
) {
    auto normals_buf = face_normals.unchecked<2>();
    auto faces_buf = faces.unchecked<2>();
    auto mesh_verts_buf = mesh_vertices.unchecked<2>();
    auto tet_verts_buf = tet_vertices.unchecked<2>();
    auto tet_labels_buf = tet_boundary_labels.unchecked<1>();
    auto d1_buf = d1_arr.unchecked<1>();
    auto d2_buf = d2_arr.unchecked<1>();
    
    const int64_t n_faces = faces_buf.shape(0);
    const int64_t n_tet_verts = tet_verts_buf.shape(0);
    
    Vec3 d1 = {d1_buf(0), d1_buf(1), d1_buf(2)};
    Vec3 d2 = {d2_buf(0), d2_buf(1), d2_buf(2)};
    
    // =========================================================================
    // Step 1: Identify outer vs inner boundary using tet labels (KDTree approach)
    // =========================================================================
    
    // Build KDTree for tet vertices
    std::vector<Vec3> tet_points(n_tet_verts);
    for (int64_t i = 0; i < n_tet_verts; i++) {
        tet_points[i] = {tet_verts_buf(i, 0), tet_verts_buf(i, 1), tet_verts_buf(i, 2)};
    }
    
    // For each mesh vertex, find nearest tet vertex
    const int64_t n_mesh_verts = mesh_verts_buf.shape(0);
    std::vector<int64_t> mesh_to_tet(n_mesh_verts);
    
    // Simple O(n*m) nearest neighbor - could use spatial hash for larger meshes
#pragma omp parallel for
    for (int64_t vi = 0; vi < n_mesh_verts; vi++) {
        Vec3 p = {mesh_verts_buf(vi, 0), mesh_verts_buf(vi, 1), mesh_verts_buf(vi, 2)};
        double best_dist = std::numeric_limits<double>::infinity();
        int64_t best_idx = 0;
        
        for (int64_t ti = 0; ti < n_tet_verts; ti++) {
            double dx = p[0] - tet_points[ti][0];
            double dy = p[1] - tet_points[ti][1];
            double dz = p[2] - tet_points[ti][2];
            double dist = dx*dx + dy*dy + dz*dz;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = ti;
            }
        }
        mesh_to_tet[vi] = best_idx;
    }
    
    // Get vertex labels from tet mesh
    std::vector<int8_t> vertex_labels(n_mesh_verts);
    for (int64_t vi = 0; vi < n_mesh_verts; vi++) {
        vertex_labels[vi] = tet_labels_buf(mesh_to_tet[vi]);
    }
    
    // Classify faces: inner if ALL vertices are inner (-1)
    std::vector<bool> is_outer(n_faces);
    std::vector<int64_t> outer_indices;
    outer_indices.reserve(n_faces);
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        int8_t l0 = vertex_labels[faces_buf(fi, 0)];
        int8_t l1 = vertex_labels[faces_buf(fi, 1)];
        int8_t l2 = vertex_labels[faces_buf(fi, 2)];
        
        is_outer[fi] = !(l0 == -1 && l1 == -1 && l2 == -1);
        if (is_outer[fi]) {
            outer_indices.push_back(fi);
        }
    }
    
    const int64_t n_outer = outer_indices.size();
    const int64_t n_inner = n_faces - n_outer;
    
    if (n_outer == 0) {
        // All faces are inner - return early
        CompleteMoldHalfResult result;
        result.face_labels = py::array_t<int8_t>(n_faces);
        auto labels_ptr = result.face_labels.mutable_unchecked<1>();
        for (int64_t i = 0; i < n_faces; i++) labels_ptr(i) = -1;
        
        result.face_colors = py::array_t<uint8_t>({n_faces, int64_t(4)});
        auto colors_ptr = result.face_colors.mutable_unchecked<2>();
        for (int64_t i = 0; i < n_faces; i++) {
            colors_ptr(i, 0) = 80; colors_ptr(i, 1) = 80;
            colors_ptr(i, 2) = 80; colors_ptr(i, 3) = 255;
        }
        result.n_h1 = 0; result.n_h2 = 0;
        result.n_boundary_zone = 0; result.n_inner = n_faces;
        return result;
    }
    
    // =========================================================================
    // Step 2: Build adjacency for all faces
    // =========================================================================
    
    struct PairHash {
        size_t operator()(const std::pair<int64_t, int64_t>& p) const {
            return std::hash<int64_t>()(p.first) ^ (std::hash<int64_t>()(p.second) << 1);
        }
    };
    
    std::unordered_map<std::pair<int64_t, int64_t>, std::vector<int64_t>, PairHash> edge_to_faces;
    edge_to_faces.reserve(n_faces * 3);
    
    for (int64_t fi = 0; fi < n_faces; fi++) {
        int64_t v0 = faces_buf(fi, 0);
        int64_t v1 = faces_buf(fi, 1);
        int64_t v2 = faces_buf(fi, 2);
        
        auto make_edge = [](int64_t a, int64_t b) {
            return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        };
        
        edge_to_faces[make_edge(v0, v1)].push_back(fi);
        edge_to_faces[make_edge(v1, v2)].push_back(fi);
        edge_to_faces[make_edge(v2, v0)].push_back(fi);
    }
    
    // Build adjacency lists
    std::vector<std::vector<int64_t>> adjacency(n_faces);
    for (const auto& [edge, face_list] : edge_to_faces) {
        if (face_list.size() == 2) {
            adjacency[face_list[0]].push_back(face_list[1]);
            adjacency[face_list[1]].push_back(face_list[0]);
        }
    }
    
    // =========================================================================
    // Step 3: Classify outer triangles using greedy region growing
    // =========================================================================
    
    // Create global->local mapping for outer triangles
    std::vector<int64_t> global_to_local(n_faces, -1);
    for (int64_t i = 0; i < n_outer; i++) {
        global_to_local[outer_indices[i]] = i;
    }
    
    // Compute dot products
    std::vector<double> dot1(n_outer), dot2(n_outer);
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t fi = outer_indices[i];
        Vec3 normal = {normals_buf(fi, 0), normals_buf(fi, 1), normals_buf(fi, 2)};
        dot1[i] = vec3_dot(normal, d1);
        dot2[i] = vec3_dot(normal, d2);
    }
    
    // Find seeds
    int64_t f1_local = 0, f2_local = 0;
    double best1 = -1e30, best2 = -1e30;
    for (int64_t i = 0; i < n_outer; i++) {
        if (dot1[i] > best1) { best1 = dot1[i]; f1_local = i; }
        if (dot2[i] > best2) { best2 = dot2[i]; f2_local = i; }
    }
    
    // Initialize labels
    std::vector<int8_t> outer_labels(n_outer, -1);
    outer_labels[f1_local] = 1;
    outer_labels[f2_local] = 2;
    
    // Greedy expansion with priority queues
    using PQEntry = std::pair<double, int64_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq1, pq2;
    std::vector<bool> in_queue(n_outer, false);
    
    auto add_neighbors = [&](int64_t local_idx, auto& pq, const std::vector<double>& dots) {
        int64_t global_idx = outer_indices[local_idx];
        for (int64_t neighbor : adjacency[global_idx]) {
            int64_t neighbor_local = global_to_local[neighbor];
            if (neighbor_local < 0 || outer_labels[neighbor_local] >= 0 || in_queue[neighbor_local]) continue;
            pq.emplace(-dots[neighbor_local], neighbor_local);
            in_queue[neighbor_local] = true;
        }
    };
    
    add_neighbors(f1_local, pq1, dot1);
    add_neighbors(f2_local, pq2, dot2);
    
    while (!pq1.empty() || !pq2.empty()) {
        while (!pq1.empty()) {
            auto [neg_score, local_idx] = pq1.top(); pq1.pop();
            if (outer_labels[local_idx] >= 0) continue;
            if (dot1[local_idx] >= dot2[local_idx]) {
                outer_labels[local_idx] = 1;
                add_neighbors(local_idx, pq1, dot1);
                break;
            } else {
                pq2.emplace(-dot2[local_idx], local_idx);
            }
        }
        while (!pq2.empty()) {
            auto [neg_score, local_idx] = pq2.top(); pq2.pop();
            if (outer_labels[local_idx] >= 0) continue;
            if (dot2[local_idx] >= dot1[local_idx]) {
                outer_labels[local_idx] = 2;
                add_neighbors(local_idx, pq2, dot2);
                break;
            } else {
                pq1.emplace(-dot1[local_idx], local_idx);
            }
        }
    }
    
    // Handle unassigned
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        if (outer_labels[i] < 0) {
            outer_labels[i] = (dot1[i] >= dot2[i]) ? 1 : 2;
        }
    }
    
    // Quick Laplacian smoothing (3 iterations)
    for (int iter = 0; iter < 3; iter++) {
        std::vector<int8_t> new_labels = outer_labels;
#pragma omp parallel for
        for (int64_t i = 0; i < n_outer; i++) {
            int64_t global_idx = outer_indices[i];
            int h1_count = 0, h2_count = 0;
            for (int64_t neighbor : adjacency[global_idx]) {
                int64_t nl = global_to_local[neighbor];
                if (nl >= 0) {
                    if (outer_labels[nl] == 1) h1_count++;
                    else if (outer_labels[nl] == 2) h2_count++;
                }
            }
            if (outer_labels[i] == 1 && h2_count > h1_count * 2) new_labels[i] = 2;
            else if (outer_labels[i] == 2 && h1_count > h2_count * 2) new_labels[i] = 1;
        }
        outer_labels = new_labels;
    }
    
    // Re-apply strong constraints
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        if (dot1[i] > 0.7 && dot1[i] > dot2[i] + 0.3) outer_labels[i] = 1;
        else if (dot2[i] > 0.7 && dot2[i] > dot1[i] + 0.3) outer_labels[i] = 2;
    }
    
    // =========================================================================
    // Step 4: Find interface and compute boundary zone via BFS
    // =========================================================================
    
    std::vector<bool> is_interface(n_outer, false);
    
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t global_idx = outer_indices[i];
        int8_t my_label = outer_labels[i];
        for (int64_t neighbor : adjacency[global_idx]) {
            int64_t nl = global_to_local[neighbor];
            if (nl >= 0 && outer_labels[nl] != my_label) {
                is_interface[i] = true;
                break;
            }
        }
    }
    
    // BFS from interface triangles
    std::vector<int64_t> distance(n_outer, -1);
    std::vector<int64_t> queue;
    queue.reserve(n_outer);
    
    for (int64_t i = 0; i < n_outer; i++) {
        if (is_interface[i]) {
            distance[i] = 0;
            queue.push_back(i);
        }
    }
    
    int64_t head = 0;
    while (head < (int64_t)queue.size()) {
        int64_t local_idx = queue[head++];
        int64_t dist = distance[local_idx];
        if (dist >= boundary_zone_hops) continue;
        
        int64_t global_idx = outer_indices[local_idx];
        for (int64_t neighbor : adjacency[global_idx]) {
            int64_t nl = global_to_local[neighbor];
            if (nl >= 0 && distance[nl] < 0) {
                distance[nl] = dist + 1;
                queue.push_back(nl);
            }
        }
    }
    
    // =========================================================================
    // Step 5: Build final labels and colors
    // =========================================================================
    
    // Face labels: -1=inner, 0=boundary_zone, 1=H1, 2=H2
    auto result_labels = py::array_t<int8_t>(n_faces);
    auto labels_ptr = result_labels.mutable_unchecked<1>();
    
    // Colors: H1=green, H2=orange, boundary=light gray, inner=dark gray
    auto result_colors = py::array_t<uint8_t>({n_faces, int64_t(4)});
    auto colors_ptr = result_colors.mutable_unchecked<2>();
    
    int64_t count_h1 = 0, count_h2 = 0, count_bz = 0;
    
    // Initialize all as inner (dark gray)
#pragma omp parallel for
    for (int64_t fi = 0; fi < n_faces; fi++) {
        labels_ptr(fi) = -1;
        colors_ptr(fi, 0) = 80; colors_ptr(fi, 1) = 80;
        colors_ptr(fi, 2) = 80; colors_ptr(fi, 3) = 255;
    }
    
    // Set outer triangle labels and colors
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t fi = outer_indices[i];
        bool in_boundary_zone = (distance[i] >= 0 && distance[i] <= boundary_zone_hops);
        
        if (in_boundary_zone) {
            labels_ptr(fi) = 0;  // Boundary zone
            colors_ptr(fi, 0) = 180; colors_ptr(fi, 1) = 180;
            colors_ptr(fi, 2) = 180; colors_ptr(fi, 3) = 255;
            count_bz++;
        } else if (outer_labels[i] == 1) {
            labels_ptr(fi) = 1;  // H1 green
            colors_ptr(fi, 0) = 0; colors_ptr(fi, 1) = 255;
            colors_ptr(fi, 2) = 0; colors_ptr(fi, 3) = 255;
            count_h1++;
        } else {
            labels_ptr(fi) = 2;  // H2 orange
            colors_ptr(fi, 0) = 255; colors_ptr(fi, 1) = 102;
            colors_ptr(fi, 2) = 0; colors_ptr(fi, 3) = 255;
            count_h2++;
        }
    }
    
    CompleteMoldHalfResult result;
    result.face_labels = result_labels;
    result.face_colors = result_colors;
    result.n_h1 = count_h1;
    result.n_h2 = count_h2;
    result.n_boundary_zone = count_bz;
    result.n_inner = n_inner;
    
    return result;
}

/**
 * Fast mold half classification using greedy region growing with OpenMP.
 * 
 * Algorithm from paper:
 * 1. Find seed faces F1 and F2 with best alignment to d1 and d2
 * 2. Greedy region-growing from seeds, assigning faces based on normal alignment
 * 
 * @param face_normals (F, 3) array of face normals
 * @param outer_indices (O,) array of outer triangle indices to classify
 * @param d1, d2 Parting directions (normalized)
 * @param adj_indices, adj_ptr CSR-format adjacency
 * @return MoldHalfClassificationResult with labels
 */
MoldHalfClassificationResult classify_mold_halves_cpp(
    py::array_t<double> face_normals,
    py::array_t<int64_t> outer_indices,
    py::array_t<double> d1_arr,
    py::array_t<double> d2_arr,
    py::array_t<int64_t> adj_indices_arr,
    py::array_t<int64_t> adj_ptr_arr,
    int64_t n_total_faces
) {
    auto normals_buf = face_normals.unchecked<2>();
    auto outer_buf = outer_indices.unchecked<1>();
    auto d1_buf = d1_arr.unchecked<1>();
    auto d2_buf = d2_arr.unchecked<1>();
    auto adj_idx_buf = adj_indices_arr.unchecked<1>();
    auto adj_ptr_buf = adj_ptr_arr.unchecked<1>();
    
    const int64_t n_outer = outer_buf.shape(0);
    
    if (n_outer == 0) {
        MoldHalfClassificationResult result;
        result.side_labels = py::array_t<int8_t>(0);
        result.outer_indices = py::array_t<int64_t>(0);
        result.n_h1 = 0;
        result.n_h2 = 0;
        return result;
    }
    
    // Extract directions
    Vec3 d1 = {d1_buf(0), d1_buf(1), d1_buf(2)};
    Vec3 d2 = {d2_buf(0), d2_buf(1), d2_buf(2)};
    
    // Create mapping from global face index to local index (position in outer_indices)
    std::vector<int64_t> global_to_local(n_total_faces, -1);
    for (int64_t i = 0; i < n_outer; i++) {
        global_to_local[outer_buf(i)] = i;
    }
    
    // Compute dot products with parting directions (parallel)
    std::vector<double> dot1(n_outer), dot2(n_outer);
    
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t fi = outer_buf(i);
        Vec3 normal = {normals_buf(fi, 0), normals_buf(fi, 1), normals_buf(fi, 2)};
        dot1[i] = vec3_dot(normal, d1);
        dot2[i] = vec3_dot(normal, d2);
    }
    
    // Find seed faces
    int64_t f1_local = 0, f2_local = 0;
    double best1 = -std::numeric_limits<double>::infinity();
    double best2 = -std::numeric_limits<double>::infinity();
    
    for (int64_t i = 0; i < n_outer; i++) {
        if (dot1[i] > best1) {
            best1 = dot1[i];
            f1_local = i;
        }
        if (dot2[i] > best2) {
            best2 = dot2[i];
            f2_local = i;
        }
    }
    
    // Initialize labels: -1 = unassigned, 1 = H1, 2 = H2
    std::vector<int8_t> labels(n_outer, -1);
    labels[f1_local] = 1;
    labels[f2_local] = 2;
    
    // Priority queues for greedy expansion: (-alignment_score, local_idx)
    using PQEntry = std::pair<double, int64_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq1, pq2;
    
    std::vector<bool> in_queue(n_outer, false);
    
    // Helper to add neighbors to queues
    auto add_neighbors_to_queue = [&](int64_t local_idx, int8_t side, 
                                       std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>>& pq,
                                       const Vec3& dir, const std::vector<double>& dots) {
        int64_t global_idx = outer_buf(local_idx);
        int64_t start = adj_ptr_buf(global_idx);
        int64_t end = adj_ptr_buf(global_idx + 1);
        
        for (int64_t j = start; j < end; j++) {
            int64_t neighbor_global = adj_idx_buf(j);
            int64_t neighbor_local = global_to_local[neighbor_global];
            
            if (neighbor_local < 0) continue;  // Not in outer set
            if (labels[neighbor_local] >= 0) continue;  // Already assigned
            if (in_queue[neighbor_local]) continue;  // Already queued
            
            double score = dots[neighbor_local];
            pq.emplace(-score, neighbor_local);  // Negative for max-heap behavior
            in_queue[neighbor_local] = true;
        }
    };
    
    // Add seed neighbors
    add_neighbors_to_queue(f1_local, 1, pq1, d1, dot1);
    add_neighbors_to_queue(f2_local, 2, pq2, d2, dot2);
    
    // Greedy expansion
    while (!pq1.empty() || !pq2.empty()) {
        // Try H1 expansion
        while (!pq1.empty()) {
            auto [neg_score, local_idx] = pq1.top();
            pq1.pop();
            
            if (labels[local_idx] >= 0) continue;
            
            // Assign to H1 if better aligned with d1
            if (dot1[local_idx] >= dot2[local_idx]) {
                labels[local_idx] = 1;
                add_neighbors_to_queue(local_idx, 1, pq1, d1, dot1);
                break;
            } else {
                // Better for H2, move to that queue
                pq2.emplace(-dot2[local_idx], local_idx);
            }
        }
        
        // Try H2 expansion
        while (!pq2.empty()) {
            auto [neg_score, local_idx] = pq2.top();
            pq2.pop();
            
            if (labels[local_idx] >= 0) continue;
            
            // Assign to H2 if better aligned with d2
            if (dot2[local_idx] >= dot1[local_idx]) {
                labels[local_idx] = 2;
                add_neighbors_to_queue(local_idx, 2, pq2, d2, dot2);
                break;
            } else {
                // Better for H1, move to that queue
                pq1.emplace(-dot1[local_idx], local_idx);
            }
        }
    }
    
    // Handle any remaining unassigned (disconnected components)
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        if (labels[i] < 0) {
            labels[i] = (dot1[i] >= dot2[i]) ? 1 : 2;
        }
    }
    
    // =========================================================================
    // MORPHOLOGICAL SMOOTHING (vectorized with OpenMP)
    // =========================================================================
    
    // Build local adjacency for outer triangles only
    std::vector<std::vector<int64_t>> local_adj(n_outer);
    
    for (int64_t i = 0; i < n_outer; i++) {
        int64_t global_idx = outer_buf(i);
        int64_t start = adj_ptr_buf(global_idx);
        int64_t end = adj_ptr_buf(global_idx + 1);
        
        for (int64_t j = start; j < end; j++) {
            int64_t neighbor_global = adj_idx_buf(j);
            int64_t neighbor_local = global_to_local[neighbor_global];
            if (neighbor_local >= 0) {
                local_adj[i].push_back(neighbor_local);
            }
        }
    }
    
    // Helper: count neighbors with specific label (parallel)
    auto count_neighbors_with_label = [&](const std::vector<int8_t>& lbl, int8_t target) {
        std::vector<int32_t> counts(n_outer, 0);
#pragma omp parallel for
        for (int64_t i = 0; i < n_outer; i++) {
            int32_t cnt = 0;
            for (int64_t neighbor : local_adj[i]) {
                if (lbl[neighbor] == target) cnt++;
            }
            counts[i] = cnt;
        }
        return counts;
    };
    
    auto get_neighbor_counts = [&]() {
        std::vector<int32_t> counts(n_outer, 0);
#pragma omp parallel for
        for (int64_t i = 0; i < n_outer; i++) {
            counts[i] = static_cast<int32_t>(local_adj[i].size());
        }
        return counts;
    };
    
    std::vector<int32_t> neighbor_counts = get_neighbor_counts();
    
    // Morphological opening: erode H1, dilate H1, erode H2, dilate H2
    for (int morph_pass = 0; morph_pass < 4; morph_pass++) {
        int8_t target_side = (morph_pass < 2) ? 1 : 2;
        int8_t flip_to = (target_side == 1) ? 2 : 1;
        
        for (int iter = 0; iter < 4; iter++) {
            auto opp_counts = count_neighbors_with_label(labels, flip_to);
            
            std::vector<bool> flip_mask(n_outer, false);
#pragma omp parallel for
            for (int64_t i = 0; i < n_outer; i++) {
                if (labels[i] == target_side && neighbor_counts[i] > 0 &&
                    opp_counts[i] > 0 && opp_counts[i] >= neighbor_counts[i] / 2) {
                    flip_mask[i] = true;
                }
            }
            
            bool any_flip = false;
            for (int64_t i = 0; i < n_outer; i++) {
                if (flip_mask[i]) {
                    labels[i] = flip_to;
                    any_flip = true;
                }
            }
            
            if (!any_flip) break;
        }
        
        // After erode, swap for dilate
        if (morph_pass == 0 || morph_pass == 2) {
            std::swap(target_side, flip_to);
        }
    }
    
    // Laplacian smoothing: 10 passes
    for (int smooth_iter = 0; smooth_iter < 10; smooth_iter++) {
        auto h1_counts = count_neighbors_with_label(labels, 1);
        auto h2_counts = count_neighbors_with_label(labels, 2);
        
        std::vector<bool> flip_to_h1(n_outer, false);
        std::vector<bool> flip_to_h2(n_outer, false);
        
#pragma omp parallel for
        for (int64_t i = 0; i < n_outer; i++) {
            if (neighbor_counts[i] == 0) continue;
            
            if (labels[i] == 1 && h2_counts[i] > h1_counts[i] * 3 / 2) {
                flip_to_h2[i] = true;
            } else if (labels[i] == 2 && h1_counts[i] > h2_counts[i] * 3 / 2) {
                flip_to_h1[i] = true;
            }
        }
        
        bool any_flip = false;
        for (int64_t i = 0; i < n_outer; i++) {
            if (flip_to_h1[i]) { labels[i] = 1; any_flip = true; }
            if (flip_to_h2[i]) { labels[i] = 2; any_flip = true; }
        }
        
        if (!any_flip) break;
    }
    
    // Re-apply strong directional constraints
    constexpr double strong_threshold = 0.7;
#pragma omp parallel for
    for (int64_t i = 0; i < n_outer; i++) {
        if (dot1[i] > strong_threshold && dot1[i] > dot2[i] + 0.3) {
            labels[i] = 1;
        } else if (dot2[i] > strong_threshold && dot2[i] > dot1[i] + 0.3) {
            labels[i] = 2;
        }
    }
    
    // Count results
    int64_t n_h1 = 0, n_h2 = 0;
    for (int64_t i = 0; i < n_outer; i++) {
        if (labels[i] == 1) n_h1++;
        else if (labels[i] == 2) n_h2++;
    }
    
    // Build result arrays
    auto result_labels = py::array_t<int8_t>(n_outer);
    auto result_labels_ptr = result_labels.mutable_unchecked<1>();
    for (int64_t i = 0; i < n_outer; i++) {
        result_labels_ptr(i) = labels[i];
    }
    
    MoldHalfClassificationResult result;
    result.side_labels = result_labels;
    result.outer_indices = outer_indices;
    result.n_h1 = n_h1;
    result.n_h2 = n_h2;
    
    return result;
}


// ============================================================================
// PYBIND11 MODULE DEFINITION
// ============================================================================

PYBIND11_MODULE(fast_algorithms, m) {
    m.doc() = "Fast C++ implementations for mesh algorithms";
    
    // Dijkstra result structure
    py::class_<DijkstraResult>(m, "DijkstraResult")
        .def_readonly("escape_labels", &DijkstraResult::escape_labels)
        .def_readonly("vertex_indices", &DijkstraResult::vertex_indices)
        .def_readonly("distances", &DijkstraResult::distances)
        .def_readonly("destinations", &DijkstraResult::destinations)
        .def_readonly("paths", &DijkstraResult::paths);
    
    // Dijkstra escape labeling
    m.def("dijkstra_escape_labeling", &dijkstra_escape_labeling,
          py::arg("edges"),
          py::arg("edge_lengths"),
          py::arg("edge_weights"),
          py::arg("boundary_labels"),
          py::arg("use_weighted_edges") = true,
          R"doc(
          Fast Dijkstra escape labeling from interior vertices to H1/H2 boundaries.
          
          For each interior vertex, finds the shortest weighted path to either H1 or H2
          boundary and labels the vertex accordingly.
          
          Parameters
          ----------
          edges : ndarray of shape (E, 2)
              Edge vertex indices
          edge_lengths : ndarray of shape (E,)
              Edge lengths
          edge_weights : ndarray of shape (E,)
              Edge weights (higher = lower cost near part)
          boundary_labels : ndarray of shape (N,)
              Vertex labels: 0=interior, 1=H1, 2=H2, -1=inner boundary
          use_weighted_edges : bool
              If True, use edge_length * edge_weight as cost
              
          Returns
          -------
          DijkstraResult
              Contains escape_labels, vertex_indices, distances, destinations, paths
          )doc");
    
    // Edge boundary labels
    m.def("compute_edge_boundary_labels", &compute_edge_boundary_labels,
          py::arg("edges"),
          py::arg("boundary_labels"),
          py::arg("boundary_faces"),
          py::arg("boundary_mesh_vertices"),
          py::arg("tet_vertices"),
          R"doc(
          Compute edge boundary labels based on vertex labels and boundary surface membership.
          
          Parameters
          ----------
          edges : ndarray of shape (E, 2)
              Edge vertex indices
          boundary_labels : ndarray of shape (N,)
              Vertex labels: 0=interior, 1=H1, 2=H2, -1=inner
          boundary_faces : ndarray of shape (F, 3)
              Boundary mesh face vertex indices
          boundary_mesh_vertices : ndarray of shape (B, 3)
              Boundary mesh vertex positions
          tet_vertices : ndarray of shape (N, 3)
              Tetrahedral mesh vertex positions
              
          Returns
          -------
          ndarray of shape (E,)
              Edge labels: 0=interior, 1=H1, 2=H2, -1=inner, -2=mixed
          )doc");
    
    // Boundary to tet mapping
    m.def("build_boundary_to_tet_mapping", &build_boundary_to_tet_mapping,
          py::arg("boundary_vertices"),
          py::arg("tet_vertices"),
          R"doc(
          Build mapping from boundary mesh vertices to tetrahedral mesh vertices.
          
          Parameters
          ----------
          boundary_vertices : ndarray of shape (B, 3)
              Boundary mesh vertex positions
          tet_vertices : ndarray of shape (N, 3)
              Tetrahedral mesh vertex positions
              
          Returns
          -------
          ndarray of shape (B,)
              Index of nearest tet vertex for each boundary vertex
          )doc");
    
    // Secondary cuts result structure
    py::class_<SecondaryCutsResult>(m, "SecondaryCutsResult")
        .def_readonly("cutting_edges", &SecondaryCutsResult::cutting_edges)
        .def_readonly("n_membranes_checked", &SecondaryCutsResult::n_membranes_checked)
        .def_readonly("n_intersections_found", &SecondaryCutsResult::n_intersections_found);
    
    // Secondary cuts detection
    m.def("find_secondary_cuts", &find_secondary_cuts_cpp,
          py::arg("membrane_edge_vi"),
          py::arg("membrane_edge_vj"),
          py::arg("membrane_path_i"),
          py::arg("membrane_path_i_offsets"),
          py::arg("membrane_path_j"),
          py::arg("membrane_path_j_offsets"),
          py::arg("membrane_boundary_path"),
          py::arg("membrane_boundary_offsets"),
          py::arg("seed_triangles"),
          py::arg("seed_triangle_vertices"),
          py::arg("tet_vertices"),
          py::arg("boundary_vertices"),
          py::arg("min_intersection_count") = 20,
          py::arg("min_membrane_thickness") = 0.0,
          R"doc(
          Find secondary cutting edges using C++ with OpenMP parallelization.
          
          Checks if membranes between same-label interior vertices intersect
          the part mesh (represented by seed triangles).
          
          Uses boolean intersection detection with minimum count threshold.
          
          Parameters
          ----------
          membrane_edge_vi : ndarray of shape (M,)
              First vertex index for each membrane edge
          membrane_edge_vj : ndarray of shape (M,)
              Second vertex index for each membrane edge
          membrane_path_i : ndarray
              Flattened array of path_i vertex indices
          membrane_path_i_offsets : ndarray of shape (M,)
              Offset into path_i array for each membrane
          membrane_path_j : ndarray
              Flattened array of path_j vertex indices  
          membrane_path_j_offsets : ndarray of shape (M,)
              Offset into path_j array for each membrane
          membrane_boundary_path : ndarray
              Flattened array of boundary path indices
          membrane_boundary_offsets : ndarray of shape (M,)
              Offset into boundary_path array for each membrane
          seed_triangles : ndarray of shape (T, 3, 3)
              Seed triangle vertex positions
          seed_triangle_vertices : ndarray of shape (T, 3)
              Seed triangle vertex indices (for skip checking)
          tet_vertices : ndarray of shape (N, 3)
              Tetrahedral mesh vertex positions
          boundary_vertices : ndarray of shape (B, 3)
              Boundary mesh vertex positions
          min_intersection_count : int
              Minimum number of segment-triangle intersections required (1-50)
          min_membrane_thickness : float
              Skip membranes thinner than this
              
          Returns
          -------
          SecondaryCutsResult
              Contains cutting_edges list and statistics
          )doc");
    
    // Mold half classification result structure
    py::class_<MoldHalfClassificationResult>(m, "MoldHalfClassificationResult")
        .def_readonly("side_labels", &MoldHalfClassificationResult::side_labels)
        .def_readonly("outer_indices", &MoldHalfClassificationResult::outer_indices)
        .def_readonly("n_h1", &MoldHalfClassificationResult::n_h1)
        .def_readonly("n_h2", &MoldHalfClassificationResult::n_h2);
    
    // Mold half classification
    m.def("classify_mold_halves", &classify_mold_halves_cpp,
          py::arg("face_normals"),
          py::arg("outer_indices"),
          py::arg("d1"),
          py::arg("d2"),
          py::arg("adj_indices"),
          py::arg("adj_ptr"),
          py::arg("n_total_faces"),
          R"doc(
          Fast mold half classification using greedy region growing with OpenMP.
          
          Classifies outer boundary triangles into H1 and H2 mold halves based on
          alignment with parting directions d1 and d2.
          
          Parameters
          ----------
          face_normals : ndarray of shape (F, 3)
              Face normals for all triangles
          outer_indices : ndarray of shape (O,)
              Indices of outer boundary triangles to classify
          d1 : ndarray of shape (3,)
              First parting direction (normalized)
          d2 : ndarray of shape (3,)
              Second parting direction (normalized)
          adj_indices : ndarray
              CSR adjacency indices
          adj_ptr : ndarray of shape (F+1,)
              CSR adjacency pointers
          n_total_faces : int
              Total number of faces in mesh
              
          Returns
          -------
          MoldHalfClassificationResult
              Contains side_labels (1=H1, 2=H2), outer_indices, n_h1, n_h2
          )doc");
    
    // Optimized mold half classification (builds adjacency internally)
    m.def("classify_mold_halves_from_faces", &classify_mold_halves_from_faces_cpp,
          py::arg("face_normals"),
          py::arg("faces"),
          py::arg("outer_indices"),
          py::arg("d1"),
          py::arg("d2"),
          R"doc(
          Fast mold half classification - builds adjacency from faces internally.
          
          This is faster than classify_mold_halves as it avoids Python overhead
          for adjacency construction.
          
          Parameters
          ----------
          face_normals : ndarray of shape (F, 3)
              Face normals for all triangles
          faces : ndarray of shape (F, 3)
              Face vertex indices
          outer_indices : ndarray of shape (O,)
              Indices of outer boundary triangles to classify
          d1 : ndarray of shape (3,)
              First parting direction (normalized)
          d2 : ndarray of shape (3,)
              Second parting direction (normalized)
              
          Returns
          -------
          MoldHalfClassificationResult
              Contains side_labels (1=H1, 2=H2), outer_indices, n_h1, n_h2
          )doc");
    
    // Complete mold half classification result
    py::class_<CompleteMoldHalfResult>(m, "CompleteMoldHalfResult")
        .def_readonly("face_labels", &CompleteMoldHalfResult::face_labels)
        .def_readonly("face_colors", &CompleteMoldHalfResult::face_colors)
        .def_readonly("n_h1", &CompleteMoldHalfResult::n_h1)
        .def_readonly("n_h2", &CompleteMoldHalfResult::n_h2)
        .def_readonly("n_boundary_zone", &CompleteMoldHalfResult::n_boundary_zone)
        .def_readonly("n_inner", &CompleteMoldHalfResult::n_inner);
    
    // Complete mold half classification pipeline
    m.def("classify_mold_halves_complete", &classify_mold_halves_complete_cpp,
          py::arg("face_normals"),
          py::arg("faces"),
          py::arg("mesh_vertices"),
          py::arg("tet_vertices"),
          py::arg("tet_boundary_labels"),
          py::arg("d1"),
          py::arg("d2"),
          py::arg("boundary_zone_hops"),
          R"doc(
          Complete mold half classification pipeline in C++.
          
          Handles everything: outer boundary detection, classification,
          boundary zone computation, and color generation.
          
          Parameters
          ----------
          face_normals : ndarray of shape (F, 3)
              Face normals for all triangles
          faces : ndarray of shape (F, 3)
              Face vertex indices
          mesh_vertices : ndarray of shape (V, 3)
              Mesh vertex positions
          tet_vertices : ndarray of shape (T, 3)
              Tetrahedral mesh vertex positions
          tet_boundary_labels : ndarray of shape (T,)
              Boundary labels: -1=inner, 1=H1, 2=H2, 0=interior
          d1 : ndarray of shape (3,)
              First parting direction
          d2 : ndarray of shape (3,)
              Second parting direction
          boundary_zone_hops : int
              BFS expansion hops for boundary zone
              
          Returns
          -------
          CompleteMoldHalfResult
              Contains face_labels, face_colors, counts
          )doc");
}
