"""
Pattern Inversion Analyzer for Seasonal Strategies

This module analyzes where actual seasonal patterns deviate from theoretical expectations,
identifying opportunities for contrarian seasonal strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from seasonality_engine import CommoditySeasonalityAnalyzer
from seasonality_strategies import (
    EnergySeasonalStrategy, AgriculturalSeasonalStrategy, 
    MetalsSeasonalStrategy, SeasonalPosition
)
import warnings
warnings.filterwarnings('ignore')


class PatternInversionAnalyzer:
    """
    Analyzes seasonal pattern inversions and identifies contrarian opportunities.
    
    Compares theoretical seasonal expectations with actual observed patterns
    to detect where traditional seasonal logic has been inverted.
    """
    
    def __init__(self):
        self.analyzer = CommoditySeasonalityAnalyzer()
        self.theoretical_expectations = self._define_theoretical_expectations()
        
    def _define_theoretical_expectations(self) -> Dict:
        """Define what we theoretically expect from seasonal patterns"""
        return {
            # Energy - Traditional heating/cooling expectations
            'CL_F': {
                'strong_months': [5, 6, 7, 8],  # Summer driving season
                'weak_months': [2, 3, 4],       # Spring maintenance
                'rationale': 'Summer driving season demand'
            },
            'NG_F': {
                'strong_months': [11, 12, 1, 2],  # Winter heating
                'weak_months': [4, 5, 6, 7],      # Summer low demand
                'rationale': 'Winter heating demand'
            },
            'HO_F': {
                'strong_months': [10, 11, 12, 1, 2],  # Winter heating
                'weak_months': [5, 6, 7, 8],           # Summer low demand
                'rationale': 'Winter heating oil demand'
            },
            'RB_F': {
                'strong_months': [3, 4, 5, 6],  # Driving season prep
                'weak_months': [9, 10, 11],     # Post-summer
                'rationale': 'Driving season preparation'
            },
            
            # Metals - Traditional calendar effects
            'GC_F': {
                'strong_months': [12, 1],  # January effect
                'weak_months': [4, 5],     # May weakness
                'rationale': 'January effect and year-end positioning'
            },
            'SI_F': {
                'strong_months': [12, 1],  # January effect
                'weak_months': [4, 5],     # May weakness
                'rationale': 'January effect for precious metals'
            },
            'HG_F': {
                'strong_months': [1, 2, 3],  # Chinese New Year restart
                'weak_months': [7, 8, 9],    # Summer industrial slowdown
                'rationale': 'Industrial demand cycles'
            },
            
            # Agriculture - Traditional harvest cycles
            'ZC_F': {
                'strong_months': [11, 12, 1],  # Post-harvest inventory
                'weak_months': [7, 8, 9],      # Harvest pressure
                'rationale': 'Corn harvest cycle'
            },
            'ZW_F': {
                'strong_months': [9, 10, 11],  # Winter wheat planting
                'weak_months': [5, 6, 7],      # Harvest season
                'rationale': 'Wheat planting and harvest cycles'
            },
            'ZS_F': {
                'strong_months': [11, 12, 1],  # Post-harvest
                'weak_months': [8, 9, 10],     # Harvest pressure
                'rationale': 'Soybean harvest cycle'
            },
            'SB_F': {
                'strong_months': [9, 10, 11],  # Brazilian crush season
                'weak_months': [2, 3, 4],      # Off-season
                'rationale': 'Sugar crush season in Brazil'
            }
        }
    
    def analyze_pattern_inversions(self, returns_data: pd.DataFrame) -> Dict:
        """
        Analyze where actual patterns contradict theoretical expectations.
        
        Returns:
            Dictionary with inversion analysis for each commodity
        """
        print("Analyzing pattern inversions...")
        
        # Calculate monthly seasonality
        seasonal_stats = self.analyzer.get_seasonal_summary_stats(returns_data)
        monthly_returns = seasonal_stats['monthly_returns']
        
        inversion_results = {}
        
        for commodity in self.theoretical_expectations.keys():
            if commodity not in monthly_returns.columns:
                continue
                
            expectation = self.theoretical_expectations[commodity]
            actual_returns = monthly_returns[commodity]
            
            # Convert month names to numbers for comparison
            month_mapping = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            
            # Calculate actual performance in expected strong/weak periods
            strong_months_performance = []
            weak_months_performance = []
            
            for month_name, month_return in actual_returns.items():
                month_num = month_mapping[month_name]
                if month_num in expectation['strong_months']:
                    strong_months_performance.append(month_return)
                elif month_num in expectation['weak_months']:
                    weak_months_performance.append(month_return)
            
            avg_strong_performance = np.mean(strong_months_performance) if strong_months_performance else 0
            avg_weak_performance = np.mean(weak_months_performance) if weak_months_performance else 0
            
            # Check for inversion
            expected_spread = avg_strong_performance - avg_weak_performance
            is_inverted = expected_spread < 0  # Strong periods performing worse than weak
            
            inversion_results[commodity] = {
                'expected_strong_months': expectation['strong_months'],
                'expected_weak_months': expectation['weak_months'],
                'actual_strong_performance': avg_strong_performance,
                'actual_weak_performance': avg_weak_performance,
                'expected_spread': expected_spread,
                'is_inverted': is_inverted,
                'inversion_magnitude': abs(expected_spread),
                'rationale': expectation['rationale'],
                'sector': self.analyzer.sector_mapping.get(commodity, 'Unknown')
            }
        
        return inversion_results
    
    def generate_inversion_report(self, inversion_results: Dict) -> pd.DataFrame:
        """Generate a comprehensive inversion analysis report"""
        
        report_data = []
        for commodity, result in inversion_results.items():
            report_data.append({
                'Commodity': commodity,
                'Sector': result['sector'],
                'Theoretical_Rationale': result['rationale'],
                'Expected_Strong_Performance_%': result['actual_strong_performance'] * 100,
                'Expected_Weak_Performance_%': result['actual_weak_performance'] * 100,
                'Pattern_Spread_%': result['expected_spread'] * 100,
                'Is_Inverted': result['is_inverted'],
                'Inversion_Magnitude_%': result['inversion_magnitude'] * 100,
                'Contrarian_Opportunity': 'HIGH' if result['is_inverted'] and result['inversion_magnitude'] > 0.005 else 'LOW'
            })
        
        return pd.DataFrame(report_data).set_index('Commodity')
    
    def plot_inversion_analysis(self, inversion_results: Dict, returns_data: pd.DataFrame):
        """Create comprehensive visualization of pattern inversions"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Inversion magnitude by commodity
        commodities = list(inversion_results.keys())
        magnitudes = [inversion_results[c]['inversion_magnitude'] * 100 for c in commodities]
        is_inverted = [inversion_results[c]['is_inverted'] for c in commodities]
        
        colors = ['red' if inv else 'green' for inv in is_inverted]
        bars = ax1.barh(commodities, magnitudes, color=colors, alpha=0.7)
        ax1.set_xlabel('Inversion Magnitude (%)')
        ax1.set_title('Pattern Inversion Analysis by Commodity\n(Red = Inverted, Green = Normal)', 
                     fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add sector labels
        for i, commodity in enumerate(commodities):
            sector = inversion_results[commodity]['sector']
            ax1.text(magnitudes[i] + 0.1, i, f"({sector})", va='center', fontsize=8)
        
        # Plot 2: Sector-level inversion summary
        sector_inversions = {}
        for commodity, result in inversion_results.items():
            sector = result['sector']
            if sector not in sector_inversions:
                sector_inversions[sector] = {'total': 0, 'inverted': 0}
            sector_inversions[sector]['total'] += 1
            if result['is_inverted']:
                sector_inversions[sector]['inverted'] += 1
        
        sectors = list(sector_inversions.keys())
        inversion_rates = [sector_inversions[s]['inverted'] / sector_inversions[s]['total'] * 100 
                          for s in sectors]
        
        bars = ax2.bar(sectors, inversion_rates, color=['orange', 'green', 'blue'], alpha=0.7)
        ax2.set_ylabel('Inversion Rate (%)')
        ax2.set_title('Pattern Inversion Rate by Sector', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, inversion_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Expected vs Actual Performance Scatter
        expected_strong = [inversion_results[c]['actual_strong_performance'] * 100 for c in commodities]
        expected_weak = [inversion_results[c]['actual_weak_performance'] * 100 for c in commodities]
        
        scatter = ax3.scatter(expected_weak, expected_strong, 
                            c=['red' if inversion_results[c]['is_inverted'] else 'green' for c in commodities],
                            s=100, alpha=0.7)
        
        # Add diagonal line (where strong = weak, i.e., no seasonal effect)
        min_val = min(min(expected_strong), min(expected_weak))
        max_val = max(max(expected_strong), max(expected_weak))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax3.set_xlabel('Performance in Expected Weak Months (%)')
        ax3.set_ylabel('Performance in Expected Strong Months (%)')
        ax3.set_title('Expected Strong vs Weak Month Performance\n(Red = Inverted Pattern)', 
                     fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add commodity labels
        for i, commodity in enumerate(commodities):
            ax3.annotate(commodity, (expected_weak[i], expected_strong[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 4: Time series of inversions (rolling analysis)
        # Calculate 2-year rolling inversion rates
        monthly_data = returns_data.copy()
        monthly_data['Year'] = monthly_data.index.year
        
        yearly_inversions = []
        years = range(monthly_data['Year'].min() + 1, monthly_data['Year'].max())
        
        for year in years:
            year_data = monthly_data[monthly_data['Year'] <= year].iloc[-504:]  # Last 2 years
            if len(year_data) < 500:  # Need sufficient data
                continue
                
            year_inversion_results = self.analyze_pattern_inversions(year_data.drop('Year', axis=1))
            inversion_count = sum(1 for r in year_inversion_results.values() if r['is_inverted'])
            total_count = len(year_inversion_results)
            inversion_rate = inversion_count / total_count * 100 if total_count > 0 else 0
            yearly_inversions.append((year, inversion_rate))
        
        if yearly_inversions:
            years_list, rates_list = zip(*yearly_inversions)
            ax4.plot(years_list, rates_list, marker='o', linewidth=2, markersize=6)
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Pattern Inversion Rate (%)')
            ax4.set_title('Evolution of Pattern Inversions Over Time\n(2-Year Rolling Window)', 
                         fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('../results/pattern_inversion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_contrarian_opportunities(self, inversion_results: Dict) -> List[str]:
        """Identify commodities with strongest contrarian opportunities"""
        
        contrarian_opportunities = []
        
        for commodity, result in inversion_results.items():
            if (result['is_inverted'] and 
                result['inversion_magnitude'] > 0.005):  # At least 0.5% monthly difference
                
                contrarian_opportunities.append({
                    'commodity': commodity,
                    'sector': result['sector'],
                    'magnitude': result['inversion_magnitude'],
                    'rationale': result['rationale']
                })
        
        # Sort by magnitude
        contrarian_opportunities.sort(key=lambda x: x['magnitude'], reverse=True)
        return contrarian_opportunities
    
    def generate_contrarian_strategy_recommendations(self, inversion_results: Dict) -> Dict:
        """Generate specific recommendations for contrarian seasonal strategies"""
        
        recommendations = {
            'Energy': [],
            'Metals': [],
            'Agriculture': []
        }
        
        for commodity, result in inversion_results.items():
            if not result['is_inverted']:
                continue
                
            sector = result['sector']
            
            # Generate inverted position recommendations
            strong_months = result['expected_strong_months']
            weak_months = result['expected_weak_months']
            
            # Invert the logic: short during expected strong months, long during expected weak
            contrarian_recommendation = {
                'commodity': commodity,
                'short_months': strong_months,  # Short when traditionally strong
                'long_months': weak_months,     # Long when traditionally weak
                'inversion_magnitude': result['inversion_magnitude'],
                'original_rationale': result['rationale'],
                'contrarian_rationale': f"Inverted {result['rationale']} - traditional pattern has reversed"
            }
            
            recommendations[sector].append(contrarian_recommendation)
        
        return recommendations


def main():
    """Main execution function for pattern inversion analysis"""
    print("=== SEASONAL PATTERN INVERSION ANALYSIS ===\n")
    
    # Initialize analyzer
    analyzer = PatternInversionAnalyzer()
    
    # Load data
    analyzer.analyzer.load_commodity_data()
    returns_data = analyzer.analyzer.extract_returns_data()
    
    print(f"Analyzing data from {returns_data.index.min().strftime('%Y-%m-%d')} to {returns_data.index.max().strftime('%Y-%m-%d')}")
    print(f"Total observations: {len(returns_data):,} days\n")
    
    # Analyze inversions
    inversion_results = analyzer.analyze_pattern_inversions(returns_data)
    
    # Generate report
    print("PATTERN INVERSION ANALYSIS RESULTS")
    print("=" * 50)
    
    inversion_report = analyzer.generate_inversion_report(inversion_results)
    print(inversion_report)
    
    # Identify contrarian opportunities
    print(f"\n🔄 CONTRARIAN OPPORTUNITIES")
    print("-" * 30)
    
    contrarian_ops = analyzer.identify_contrarian_opportunities(inversion_results)
    for i, opp in enumerate(contrarian_ops, 1):
        print(f"{i}. {opp['commodity']} ({opp['sector']}): {opp['magnitude']*100:.2f}% inversion")
        print(f"   Original logic: {opp['rationale']}")
        print()
    
    # Generate strategy recommendations
    recommendations = analyzer.generate_contrarian_strategy_recommendations(inversion_results)
    
    print(f"\n📋 CONTRARIAN STRATEGY RECOMMENDATIONS")
    print("-" * 40)
    
    for sector, recs in recommendations.items():
        if recs:
            print(f"\n{sector.upper()} SECTOR:")
            for rec in recs:
                print(f"  • {rec['commodity']}: Short months {rec['short_months']}, Long months {rec['long_months']}")
                print(f"    Rationale: {rec['contrarian_rationale']}")
    
    # Create visualizations
    print(f"\nGenerating inversion analysis charts...")
    analyzer.plot_inversion_analysis(inversion_results, returns_data)
    
    print(f"\n✅ Pattern inversion analysis complete!")
    print("Check the results/ directory for visualizations.")
    
    return inversion_results, inversion_report, recommendations


if __name__ == "__main__":
    results = main()