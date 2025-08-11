#!/usr/bin/env python3
"""
TSMOM Grid Search Optimization
Ottimizza parametri strategia TSMOM e salva risultati in approaches/time_series/results/
"""

import pandas as pd
import numpy as np
import itertools
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

from tsmom_strategy import TSMOMStrategy

class TSMOMOptimizer:
    """Ottimizzatore TSMOM con grid search."""
    
    def __init__(self):
        self.results_dir = "results/optimization"
        os.makedirs(self.results_dir, exist_ok=True)
        self.results = []
    
    def get_parameter_grid(self) -> Dict[str, List]:
        """Definisce griglia parametri ottimizzazione."""
        return {
            'lookback_months': [6, 9, 12, 15, 18],
            'target_volatility': [0.30, 0.40, 0.50, 0.60],
            'ewma_com': [45, 60, 90, 120]
        }
    
    def evaluate_params(self, lookback_months: int, target_volatility: float, ewma_com: int) -> Dict:
        """Valuta una combinazione di parametri."""
        try:
            # Inizializza strategia con parametri
            strategy = TSMOMStrategy(
                start_date='2000-01-01',
                target_volatility=target_volatility,
                lookback_months=lookback_months,
                data_cache_dir='data/'
            )
            
            # Modifica EWMA COM
            strategy._initialize_modules()
            strategy.volatility_estimator.center_of_mass = ewma_com
            
            # Esegui strategia (senza validazione per velocità)
            results = strategy.execute_full_strategy(validate_results=False)
            portfolio_returns = results['key_data']['portfolio_returns']
            
            # Calcola metriche performance
            total_return = (1 + portfolio_returns).prod() - 1
            n_years = len(portfolio_returns) / 12
            cagr = (1 + total_return) ** (1/n_years) - 1
            volatility = portfolio_returns.std() * np.sqrt(12)
            sharpe = (portfolio_returns.mean() * 12) / volatility if volatility > 0 else -10
            
            # Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            # Altre metriche
            win_rate = (portfolio_returns > 0).mean()
            downside_std = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(12)
            sortino = (portfolio_returns.mean() * 12) / downside_std if downside_std > 0 else 0
            
            return {
                'lookback_months': lookback_months,
                'target_volatility': target_volatility,
                'ewma_com': ewma_com,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'volatility': volatility,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'total_return': total_return,
                'n_months': len(portfolio_returns),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error {lookback_months}M/{target_volatility:.0%}/{ewma_com}: {e}")
            return {
                'lookback_months': lookback_months,
                'target_volatility': target_volatility,
                'ewma_com': ewma_com,
                'cagr': -1.0,
                'sharpe_ratio': -10.0,
                'sortino_ratio': -10.0,
                'volatility': 1.0,
                'max_drawdown': -1.0,
                'win_rate': 0.0,
                'total_return': -1.0,
                'n_months': 0,
                'success': False
            }
    
    def run_optimization(self) -> Dict:
        """Esegue grid search optimization."""
        print("🚀 TSMOM PARAMETER OPTIMIZATION")
        print("=" * 50)
        
        # Definisci griglia parametri
        grid = self.get_parameter_grid()
        combinations = list(itertools.product(*grid.values()))
        param_names = list(grid.keys())
        
        print(f"📊 Parameter Grid: {len(combinations)} combinations")
        for param, values in grid.items():
            print(f"  {param}: {values}")
        print()
        
        # Ottimizza tutte le combinazioni
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))
            print(f"[{i:2d}/{len(combinations)}] Testing: {params}")
            
            result = self.evaluate_params(**params)
            self.results.append(result)
            
            if result['success']:
                print(f"    → CAGR: {result['cagr']:6.2%}, Sharpe: {result['sharpe_ratio']:6.3f}")
            else:
                print("    → FAILED")
        
        # Trova migliori risultati
        valid_results = [r for r in self.results if r['success']]
        
        if not valid_results:
            print("\n❌ No valid results found!")
            return {}
        
        # Ordina per Sharpe ratio
        best_sharpe = max(valid_results, key=lambda x: x['sharpe_ratio'])
        best_cagr = max(valid_results, key=lambda x: x['cagr'])
        
        print(f"\n🏆 BEST RESULTS:")
        print(f"Best Sharpe: {best_sharpe['sharpe_ratio']:.3f}")
        print(f"  Params: {best_sharpe['lookback_months']}M/{best_sharpe['target_volatility']:.0%}/{best_sharpe['ewma_com']}d")
        print(f"  CAGR: {best_sharpe['cagr']:.2%}, MaxDD: {best_sharpe['max_drawdown']:.2%}")
        
        print(f"\nBest CAGR: {best_cagr['cagr']:.2%}")
        print(f"  Params: {best_cagr['lookback_months']}M/{best_cagr['target_volatility']:.0%}/{best_cagr['ewma_com']}d")
        print(f"  Sharpe: {best_cagr['sharpe_ratio']:.3f}, MaxDD: {best_cagr['max_drawdown']:.2%}")
        
        # Salva risultati
        self.save_results(valid_results, best_sharpe, best_cagr)
        
        return best_sharpe
    
    def save_results(self, valid_results: List[Dict], best_sharpe: Dict, best_cagr: Dict):
        """Salva tutti i risultati dell'ottimizzazione."""
        
        # Salva risultati completi
        df = pd.DataFrame(self.results).sort_values('sharpe_ratio', ascending=False)
        results_file = f"{self.results_dir}/optimization_results.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Full results: {results_file}")
        
        # Salva top performers
        df_valid = df[df['success'] == True].head(10)
        top_file = f"{self.results_dir}/top_performers.csv"
        df_valid.to_csv(top_file, index=False)
        print(f"💾 Top 10 results: {top_file}")
        
        # Salva migliori parametri JSON
        optimization_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_combinations_tested': len(self.results),
            'successful_combinations': len(valid_results),
            'optimization_period': '2000-2025',
            'best_sharpe_ratio': {
                'parameters': {
                    'lookback_months': best_sharpe['lookback_months'],
                    'target_volatility': best_sharpe['target_volatility'],
                    'ewma_com': best_sharpe['ewma_com']
                },
                'performance': {
                    'cagr': best_sharpe['cagr'],
                    'sharpe_ratio': best_sharpe['sharpe_ratio'],
                    'volatility': best_sharpe['volatility'],
                    'max_drawdown': best_sharpe['max_drawdown'],
                    'win_rate': best_sharpe['win_rate']
                }
            },
            'best_cagr': {
                'parameters': {
                    'lookback_months': best_cagr['lookback_months'],
                    'target_volatility': best_cagr['target_volatility'],
                    'ewma_com': best_cagr['ewma_com']
                },
                'performance': {
                    'cagr': best_cagr['cagr'],
                    'sharpe_ratio': best_cagr['sharpe_ratio'],
                    'volatility': best_cagr['volatility'],
                    'max_drawdown': best_cagr['max_drawdown'],
                    'win_rate': best_cagr['win_rate']
                }
            }
        }
        
        json_file = f"{self.results_dir}/optimization_summary.json"
        with open(json_file, 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)
        print(f"💾 Summary: {json_file}")
        
        print(f"\n✅ Optimization complete! Results saved to: {self.results_dir}/")

def main():
    """Main execution."""
    optimizer = TSMOMOptimizer()
    best_result = optimizer.run_optimization()
    
    if best_result:
        print(f"\n🎯 RECOMMENDED PARAMETERS:")
        print(f"  Lookback: {best_result['lookback_months']} months")
        print(f"  Target Vol: {best_result['target_volatility']:.0%}")
        print(f"  EWMA COM: {best_result['ewma_com']} days")

if __name__ == "__main__":
    main()