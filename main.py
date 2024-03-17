import webbrowser
import pickle
import osmnx as ox
import folium   
import networkx as nx
import matplotlib.pyplot as plt


def graph_data(place_names, display=False, intersection_focus=False, focus_ignore_edges=False, intersection_edges_quanity=3, maxspeed_fallback=25, basic_stats_toggle=True, interactive_map=False, simple=True, centrality=False, save_data=True, save_data_format='gpkg'):
    
    return_data = {}
    
    if save_data:
        save_storage = {}
    
    if interactive_map:
        # Initialize a storage for later adjusting the map view to fit all places
        all_bounds = []
        
        # Initialize an empty folium map, will be set after the first place is processed
        m = None
    
    for place_name in place_names:
        network_type = "drive"
        G = ox.graph_from_place(place_name, network_type=network_type, simplify=simple)
        G_projected = ox.project_graph(G)
        
        # Convert to GeoDataFrame
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
        
        # Get area of place
        nodes_proj = ox.graph_to_gdfs(G_projected, edges=False)
        graph_area_m = nodes_proj.unary_union.convex_hull.area

        # Define a fallback speed in km/h (e.g., 30 km/h)
        fallback_speed = maxspeed_fallback

        # Enrich the graph with estimated speed limits and travel times
        # Using a fallback speed for all edges
        G_projected = ox.add_edge_speeds(G_projected, fallback=fallback_speed)  # Adds 'speed_kph' attribute to edges
        G_projected = ox.add_edge_travel_times(G_projected)  # Adds or overwrites the 'travel_time' attribute to edges
        
        # Calculating basic stats
        basic_stats = ox.stats.basic_stats(G_projected)
        
        if basic_stats_toggle:
            return_data[f'{place_name}'] = {'graph': G, 'basic_stats': basic_stats, 'gdf_nodes': gdf_nodes, 'gdf_edges': gdf_edges}
        else:
            return_data[f'{place_name}'] = {'graph': G, 'gdf_nodes': gdf_nodes, 'gdf_edges': gdf_edges}
            
        # Identifying intersections (nodes with adjusted degree > 2)
        node_degrees = dict(G_projected.degree())
        
        storage = {}
        # Depend on wether or not to focus on intersections
        if intersection_focus == True:
            intersection_nodes = [node for node, degree in node_degrees.items() if degree >= intersection_edges_quanity*2]  # Degree > 4 in bidirectional graph

            # Extracting and printing information about intersections
            for node in intersection_nodes:
                node_data = G_projected.nodes[node]
                
                # Find edges connected to this intersection node
                edges = list(G_projected.edges(node, data=True))
                storage[f'{node}'] = ([node_data])
                storage[f'{node}'].append(edges)
        else:
            # Extracting and printing information about intersections
            for node in node_degrees:
                node_data = G_projected.nodes[node]
                
                # Find edges connected to this intersection node
                edges = list(G_projected.edges(node, data=True))
                
                storage[f'{node}'] = ([node_data])
                storage[f'{node}'].append(edges)
        
        if save_data_format == 'dict':
            save_storage[f'{place_name}'] = storage
        
        if display:
            if interactive_map == False:
                if centrality == True:
                    edge_centrality = nx.closeness_centrality(nx.line_graph(G))
                    nx.set_edge_attributes(G, edge_centrality, "edge_centrality")
                    ec = ox.plot.get_edge_colors_by_attr(G, "edge_centrality", cmap="inferno")
                    fig, ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=2, node_size=0)
                else:
                    fig, ax = plt.subplots()
                    
                    # Plot the edges on the axes
                    gdf_edges.plot(ax=ax, linewidth=1, edgecolor='#BC8F8F')

                    # Plot the nodes on the same axes
                    gdf_nodes.plot(ax=ax, markersize=10, color='red')
                    
                    plt.show()
            else:
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
                
                if m is None:
                    # Find map center from the first place's bounds to initialize map
                    bounds = gdf_nodes.total_bounds
                    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                    m = folium.Map(location=center, zoom_start=14)
                all_bounds.append(gdf_nodes.total_bounds)
                
                # Add edges with detailed geometies and popups for speed from G_projected
                for u, v, key, data in G_projected.edges(keys=True, data=True):
                    if focus_ignore_edges != True and intersection_focus == True:
                        if v in intersection_nodes:
                            info_parts = [
                                f"<strong>ID:</strong> {data.get('osmid', 'N/A')}",
                                f"<strong>Name:</strong> {data.get('name', 'N/A')}",
                                f"<strong>Speed:</strong> {data.get('speed_kph', 'N/A')} km/h",
                                f"<strong>Length:</strong> {data.get('length', 'N/A')} meters",
                                f"<strong>Highway:</strong> {data.get('highway', 'N/A')}",
                                f"<strong>Oneway:</strong> {data.get('oneway', 'N/A')}",
                                f"<strong>Reversed:</strong> {data.get('reversed', 'N/A')}",
                                f"<strong>Travel time:</strong> {data.get('travel_time', 'N/A')} seconds",
                                f"<strong>Connected Nodes:</strong> {u}, {v}",
                            ]
                            popup_text = '<br>'.join([f"<span style='font-size: 12pt;'>{part}</span>" for part in info_parts])
                            popup = folium.Popup(html=popup_text, max_width=250)
                            if (u, v, key) in edges_gdf.index:
                                geom = [(lat, lon) for lon, lat in edges_gdf.loc[(u, v, key), 'geometry'].coords]
                                folium.PolyLine(
                                    locations=geom,
                                    color="blue",
                                    weight=7,  # Adjust weight as needed
                                    popup=popup
                                ).add_to(m)
                    else:
                        info_parts = [
                            f"<strong>ID:</strong> {data.get('osmid', 'N/A')}",
                            f"<strong>Name:</strong> {data.get('name', 'N/A')}",
                            f"<strong>Speed:</strong> {data.get('speed_kph', 'N/A')} km/h",
                            f"<strong>Length:</strong> {data.get('length', 'N/A')} meters",
                            f"<strong>Highway:</strong> {data.get('highway', 'N/A')}",
                            f"<strong>Oneway:</strong> {data.get('oneway', 'N/A')}",
                            f"<strong>Reversed:</strong> {data.get('reversed', 'N/A')}",
                            f"<strong>Travel time:</strong> {data.get('travel_time', 'N/A')} seconds",
                            f"<strong>Connected Nodes:</strong> {u}, {v}",
                        ]
                        popup_text = '<br>'.join([f"<span style='font-size: 12pt;'>{part}</span>" for part in info_parts])
                        popup = folium.Popup(html=popup_text, max_width=250)
                        if (u, v, key) in edges_gdf.index:
                            geom = [(lat, lon) for lon, lat in edges_gdf.loc[(u, v, key), 'geometry'].coords]
                            folium.PolyLine(
                                locations=geom,
                                color="blue",
                                weight=7,  # Adjust weight as needed
                                popup=popup
                            ).add_to(m)
                
                for node, data in nodes_gdf.iterrows():
                    if intersection_focus == True:
                        if node in intersection_nodes:
                            degree = node_degrees[node]
                            node_color = 'lime' if degree >= intersection_edges_quanity*2 else 'red'
                            node_attrs = storage[str(node)][0]
                            edge_tuples = storage[str(node)][1]
                            osmid_list = [str(edge[2]['osmid']) for edge in edge_tuples]  # Adjust based on actual structure
                            osmids_list_str = ', '.join(osmid_list)
                            info_parts = [
                                f"<strong>ID:</strong> {node}",
                                f"<strong>Y, X:</strong> {data['y']}, {data['x']}",
                                f"<strong>Lat, Lon:</strong> {data['y']}, {data['x']}",
                                f"<strong>Connected Edges OSMID:</strong> {osmids_list_str}",
                            ]
                            popup_text = '<br>'.join([f"<span style='font-size: 12pt;'>{part}</span>" for part in info_parts])
                            popup = folium.Popup(html=popup_text, max_width=300)
                            folium.CircleMarker(
                                location=(data['y'], data['x']),
                                radius=8,  # Visibility adjustment
                                color=node_color,  # Brighter green
                                fill=True,
                                fill_color="lime",
                                popup=popup
                            ).add_to(m)
                    else:
                        if node in node_degrees:
                            degree = node_degrees[node]
                            node_color = 'lime' if degree >= intersection_edges_quanity*2 else 'red'
                            node_attrs = storage[str(node)][0]
                            edge_tuples = storage[str(node)][1]
                            osmid_list = [str(edge[2]['osmid']) for edge in edge_tuples]  # Adjust based on actual structure
                            osmids_list_str = ', '.join(osmid_list)
                            info_parts = [
                                f"<strong>ID:</strong> {node}",
                                f"<strong>Y, X:</strong> {data['y']}, {data['x']}",
                                f"<strong>Lat, Lon:</strong> {data['y']}, {data['x']}",
                                f"<strong>Connected Edges OSMID:</strong> {osmids_list_str}",
                            ]
                            popup_text = '<br>'.join([f"<span style='font-size: 12pt;'>{part}</span>" for part in info_parts])
                            popup = folium.Popup(html=popup_text, max_width=300)
                            folium.CircleMarker(
                                location=(data['y'], data['x']),
                                radius=8,  # Visibility adjustment
                                color=node_color,  # Brighter green
                                fill=True,
                                fill_color="lime",
                                popup=popup
                            ).add_to(m)
        if save_data:
            if save_data_format == 'gpkg':
                # save street network as GeoPackage to work with in GIS
                ox.save_graph_geopackage(G, filepath=f'./data/network{place_name}.gpkg')
            if save_data_format == 'graphml':
                # save street network as GraphML file to work with later in OSMnx or networkx or gephi
                ox.save_graphml(G, filepath=f'./data/network{place_name}.graphml')
    
    if save_data:
        with open(f'./dictData/{place_name}.pickle', 'wb') as handle:
            pickle.dump(return_data, handle)
    
    if interactive_map and m is not None:
        # Adjust the map to fit all places
        bounds = [min([b[0] for b in all_bounds]), min([b[1] for b in all_bounds]),
                  max([b[2] for b in all_bounds]), max([b[3] for b in all_bounds])]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        
        folium.LayerControl().add_to(m)
        
        map_filename = 'map.html'
        m.save(map_filename)
        webbrowser.open(map_filename, new=2)
    
    return return_data
    

places = [['San Francisco, California, USA']]

graph_data(place_names=places, display=False, intersection_focus=True, focus_ignore_edges=False, intersection_edges_quanity=3, maxspeed_fallback=30, basic_stats_toggle=True, interactive_map=False, simple=True, centrality=False, save_data=True, save_data_format='gpkg')

