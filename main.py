"""Main analysis pipeline for Cultural Consensus Theory model."""
from cct import data, model, analysis, visualization
import arviz as az

def main():
    """Execute the full CCT analysis workflow."""
    # Load and validate data
    plant_data = data.load_plant_data()
    
    # Create and sample model
    cct_model = model.create_model(plant_data)
    trace = model.sample_model(cct_model)
    
    # Analyze results
    competence_stats = analysis.get_competence_stats(trace)
    consensus_comparison = analysis.compare_consensus_methods(trace, plant_data)
    
    # Generate visualizations
    comp_fig = visualization.plot_competence(trace)
    cons_fig = visualization.plot_consensus(trace)
    
    # Display results
    print("\nMODEL SUMMARY")
    print(az.summary(trace, var_names=["competence", "consensus"]))
    
    print("\nCOMPETENCE ANALYSIS")
    print(f"Most competent informant: #{competence_stats['most_competent']}")
    print(f"Least competent informant: #{competence_stats['least_competent']}")
    
    print("\nCONSENSUS COMPARISON")
    print(f"CCT Consensus: {consensus_comparison['cct_consensus']}")
    print(f"Majority Vote: {consensus_comparison['majority_vote']}")
    print(f"Disagreements at items: {consensus_comparison['disagreements']}")

if __name__ == "__main__":
    main()