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

namespace py = pybind11;

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
}
