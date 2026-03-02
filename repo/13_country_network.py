"""
Country-Level Network Data Preparation
Aggregates bilateral trade data to create country-level network

Network structure:
- Nodes: Countries (RCEP members)
- Edges: Bilateral trade flows (weighted, directed)
- Node attributes: GDP, population, tariff levels
- Edge attributes: Trade value, product composition
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from config import RCEP_COUNTRIES, DATA_DIR, NETWORK_WEIGHT_THRESHOLD
from utils import save_dataframe, logger, print_data_summary

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logger.warning("networkx not installed. Run: pip install networkx")


def _standardize_bilateral_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map various column names to standard: reporter, partner, year, trade_value.
    """
    df = df.copy()
    # Reporter
    if 'reporter_iso' in df.columns and 'reporter' not in df.columns:
        df['reporter'] = df['reporter_iso']
    elif 'reporterCode' in df.columns and 'reporter' not in df.columns:
        df['reporter'] = df['reporterCode']
    # Partner
    if 'partner_iso' in df.columns and 'partner' not in df.columns:
        df['partner'] = df['partner_iso']
    elif 'partnerCode' in df.columns and 'partner' not in df.columns:
        df['partner'] = df['partnerCode']
    # Year
    if 'period' in df.columns and 'year' not in df.columns:
        df['year'] = df['period']
    elif 'date' in df.columns and 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    # Trade value (prefer export_usd for directed flow)
    if 'trade_value' not in df.columns:
        if 'export_usd' in df.columns:
            df['trade_value'] = df['export_usd']
        elif 'primaryValue' in df.columns:
            df['trade_value'] = df['primaryValue']
        elif 'value' in df.columns:
            df['trade_value'] = df['value']
    needed = ['reporter', 'partner', 'year', 'trade_value']
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    return df[needed + ([c for c in ['source'] if c in df.columns] or [])].dropna(subset=['reporter', 'partner', 'year', 'trade_value'])


def load_bilateral_trade_data() -> pd.DataFrame:
    """
    Load and combine bilateral trade data from various sources.
    Prioritizes master_bilateral_trade_2005_2024.csv when present.
    
    Returns:
        Combined DataFrame with bilateral trade flows (reporter, partner, year, trade_value)
    """
    logger.info("Loading bilateral trade data...")
    
    all_data = []
    
    # 1. Master bilateral (priority: use existing comprehensive panel)
    master_file = DATA_DIR / "master_bilateral_trade_2005_2024.csv"
    if master_file.exists():
        df = pd.read_csv(master_file)
        df['source'] = 'master'
        df_std = _standardize_bilateral_columns(df)
        if not df_std.empty:
            all_data.append(df_std)
            logger.info(f"Loaded {master_file.name}: {len(df_std)} rows")
    
    # 2. User RCEP comprehensive (if present)
    for rcep_path in [DATA_DIR / "data" / "rcep_comprehensive_data.csv", DATA_DIR / "rcep_comprehensive_data.csv"]:
        if rcep_path.exists():
            df = pd.read_csv(rcep_path)
            df['source'] = 'rcep_user'
            df_std = _standardize_bilateral_columns(df)
            if not df_std.empty:
                all_data.append(df_std)
                logger.info(f"Loaded {rcep_path.name}: {len(df_std)} rows")
            break
    
    # 3. Comtrade data
    comtrade_files = list(DATA_DIR.glob("comtrade_bilateral*.csv"))
    for filepath in comtrade_files:
        df = pd.read_csv(filepath)
        df['source'] = 'comtrade'
        df_std = _standardize_bilateral_columns(df)
        if not df_std.empty:
            all_data.append(df_std)
            logger.info(f"Loaded {filepath.name}: {len(df_std)} rows")
    
    # 4. CEPII Gravity data
    cepii_files = list(DATA_DIR.glob("cepii_gravity*.csv"))
    for filepath in cepii_files:
        df = pd.read_csv(filepath)
        df['source'] = 'cepii'
        df_std = _standardize_bilateral_columns(df)
        if not df_std.empty:
            all_data.append(df_std)
            logger.info(f"Loaded {filepath.name}: {len(df_std)} rows")
    
    # 5. OECD TiVA data
    tiva_files = list(DATA_DIR.glob("oecd_tiva*processed*.csv"))
    for filepath in tiva_files:
        df = pd.read_csv(filepath)
        df['source'] = 'tiva'
        df_std = _standardize_bilateral_columns(df)
        if not df_std.empty:
            all_data.append(df_std)
            logger.info(f"Loaded {filepath.name}: {len(df_std)} rows")
    
    if not all_data:
        logger.warning("No bilateral trade data found!")
        logger.info("Please run bilateral trade acquisition scripts first or place master_bilateral_trade_2005_2024.csv in DATA_DIR")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined {len(combined)} total observations")
    
    return combined


def aggregate_to_country_level(df: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
    """
    Aggregate trade data to country-level bilateral flows
    
    Args:
        df: Raw bilateral trade data
        year: Year to aggregate
    
    Returns:
        DataFrame with country-level bilateral trade
    """
    logger.info(f"Aggregating to country level for year {year}...")
    
    # Standardize column names based on source
    # This is simplified - actual implementation depends on data structure
    
    # Expected columns: reporter, partner, year, trade_value
    
    if 'year' in df.columns:
        df = df[df['year'] == year]
    
    # Group by reporter-partner pairs
    if 'reporter' in df.columns and 'partner' in df.columns:
        aggregated = df.groupby(['reporter', 'partner']).agg({
            'trade_value': 'sum'
        }).reset_index()
        
        logger.info(f"Aggregated to {len(aggregated)} country pairs")
        return aggregated
    
    logger.warning("Could not aggregate - missing required columns")
    return pd.DataFrame()


def create_country_network(trade_df: pd.DataFrame) -> nx.DiGraph:
    """
    Create directed network from bilateral trade data
    
    Args:
        trade_df: Aggregated country-level trade data
    
    Returns:
        NetworkX DiGraph object
    """
    if not NX_AVAILABLE:
        logger.error("NetworkX not available")
        return None
    
    logger.info("Creating country-level trade network...")
    
    G = nx.DiGraph()
    
    # Add nodes (countries)
    for code, name in RCEP_COUNTRIES.items():
        G.add_node(code, name=name)
    
    # Add edges (trade flows)
    total_trade = trade_df['trade_value'].sum()
    threshold = total_trade * NETWORK_WEIGHT_THRESHOLD
    
    for _, row in trade_df.iterrows():
        reporter = row['reporter']
        partner = row['partner']
        value = row['trade_value']
        
        # Filter small flows
        if value >= threshold:
            G.add_edge(reporter, partner, weight=value)
    
    logger.info(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def calculate_network_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Calculate network centrality and connectivity metrics
    
    Args:
        G: NetworkX graph
    
    Returns:
        DataFrame with network metrics
    """
    if not NX_AVAILABLE or G is None:
        return pd.DataFrame()
    
    logger.info("Calculating network metrics...")
    
    metrics = []
    
    # Degree centrality
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Closeness centrality
    closeness = nx.closeness_centrality(G, distance='weight')
    
    # PageRank
    pagerank = nx.pagerank(G, weight='weight')
    
    for node in G.nodes():
        metrics.append({
            'country': node,
            'country_name': RCEP_COUNTRIES.get(node, node),
            'in_degree': in_degree.get(node, 0),
            'out_degree': out_degree.get(node, 0),
            'total_degree': in_degree.get(node, 0) + out_degree.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'closeness': closeness.get(node, 0),
            'pagerank': pagerank.get(node, 0)
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values('total_degree', ascending=False)
    
    print_data_summary(metrics_df, "Country Network Metrics")
    
    return metrics_df


def export_network_files(G: nx.DiGraph, year: int):
    """
    Export network in various formats
    
    Args:
        G: NetworkX graph
        year: Year of data
    """
    if not NX_AVAILABLE or G is None:
        return
    
    logger.info("Exporting network files...")
    
    # Edge list
    edge_list = []
    for u, v, data in G.edges(data=True):
        edge_list.append({
            'source': u,
            'target': v,
            'weight': data.get('weight', 0)
        })
    
    edge_df = pd.DataFrame(edge_list)
    edge_file = DATA_DIR / f"country_network_edges_{year}.csv"
    save_dataframe(edge_df, edge_file)
    
    # Node list
    node_list = []
    for node, data in G.nodes(data=True):
        node_list.append({
            'id': node,
            'label': data.get('name', node)
        })
    
    node_df = pd.DataFrame(node_list)
    node_file = DATA_DIR / f"country_network_nodes_{year}.csv"
    save_dataframe(node_df, node_file)
    
    # GML format (for Gephi, Cytoscape)
    gml_file = DATA_DIR / f"country_network_{year}.gml"
    nx.write_gml(G, gml_file)
    logger.info(f"Saved GML to {gml_file}")
    
    # GraphML format
    graphml_file = DATA_DIR / f"country_network_{year}.graphml"
    nx.write_graphml(G, graphml_file)
    logger.info(f"Saved GraphML to {graphml_file}")


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("Country-Level Network Preparation")
    logger.info("="*60)
    
    # Load bilateral trade data
    trade_df = load_bilateral_trade_data()
    
    if trade_df.empty:
        logger.error("No trade data available")
        logger.info("Please run data acquisition scripts first:")
        logger.info("- 01_comtrade_bilateral.py")
        logger.info("- 04_cepii_gravity.py")
        return
    
    # Aggregate to country level
    year = 2022
    country_trade = aggregate_to_country_level(trade_df, year=year)
    
    if not country_trade.empty:
        # Save aggregated data
        output_file = DATA_DIR / f"bilateral_trade_country_level_{year}.csv"
        save_dataframe(country_trade, output_file)
        
        # Create network
        if NX_AVAILABLE:
            G = create_country_network(country_trade)
            
            if G:
                # Calculate metrics
                metrics = calculate_network_metrics(G)
                metrics_file = DATA_DIR / f"country_network_metrics_{year}.csv"
                save_dataframe(metrics, metrics_file)
                
                # Export network files
                export_network_files(G, year)
        else:
            logger.warning("NetworkX not available - skipping network creation")
            logger.info("Install with: pip install networkx")
    
    logger.info("\n" + "="*60)
    logger.info("Country network preparation complete")
    logger.info("="*60)


if __name__ == "__main__":
    main()
