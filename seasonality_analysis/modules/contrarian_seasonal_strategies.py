"""
Contrarian Seasonal Trading Strategies

Implementation of seasonal trading strategies with inverted logic, based on 
identified pattern inversions in commodity markets. These strategies take the 
opposite position to traditional seasonal expectations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from seasonality_strategies import BaseSeasonalStrategy, SeasonalPosition
from pattern_inversion_analyzer import PatternInversionAnalyzer
import warnings
warnings.filterwarnings('ignore')


class ContrarianEnergyStrategy(BaseSeasonalStrategy):
    """
    Contrarian energy seasonal strategy - opposite of traditional heating/cooling logic.
    
    Based on analysis showing that traditional energy seasonal patterns have inverted:
    - Natural Gas: Short during winter (traditional heating season), Long during summer
    - Heating Oil: Short during winter heating season, Long during summer
    - Gasoline: Opposite of driving season preparation
    """
    
    def __init__(self):
        super().__init__(
            name="Contrarian Energy Strategy",
            description="Energy seasonality with inverted logic - opposite of heating/cooling patterns"
        )
    
    def define_seasonal_positions(self) -> List[SeasonalPosition]:
        """Define contrarian energy seasonal positions"""
        positions = [
            # Natural Gas - CONTRARIAN: Short winter heating season
            SeasonalPosition(
                commodity='NG_F',
                direction=-1,  # SHORT during traditional heating season
                entry_month=11,  # November
                exit_month=2,    # February
                expected_return=0.15,  # Based on inversion analysis
                statistical_significance=0.05,
                economic_rationale="CONTRARIAN: Pattern shows weakness during traditional heating season",
                volatility_adjustment=0.6
            ),
            
            # Natural Gas - CONTRARIAN: Long summer low-demand season
            SeasonalPosition(
                commodity='NG_F',
                direction=1,   # LONG during traditional weak season
                entry_month=4,   # April
                exit_month=7,    # July
                expected_return=0.12,
                statistical_significance=0.06,
                economic_rationale="CONTRARIAN: Pattern shows strength during traditional weak season",
                volatility_adjustment=0.6
            ),
            
            # Heating Oil - CONTRARIAN: Short winter heating season
            SeasonalPosition(
                commodity='HO_F',
                direction=-1,  # SHORT during traditional heating season
                entry_month=10,  # October
                exit_month=2,    # February
                expected_return=0.10,
                statistical_significance=0.07,
                economic_rationale="CONTRARIAN: Heating oil weak during traditional heating season",
                volatility_adjustment=0.8
            ),
            
            # Heating Oil - CONTRARIAN: Long summer season
            SeasonalPosition(
                commodity='HO_F',
                direction=1,   # LONG during traditional weak season
                entry_month=5,   # May
                exit_month=8,    # August
                expected_return=0.08,
                statistical_significance=0.08,
                economic_rationale="CONTRARIAN: Heating oil strength during traditional weak season",
                volatility_adjustment=0.8
            ),
            
            # Gasoline - CONTRARIAN: Short driving season prep
            SeasonalPosition(
                commodity='RB_F',
                direction=-1,  # SHORT during traditional driving season prep
                entry_month=3,   # March
                exit_month=6,    # June
                expected_return=0.06,
                statistical_significance=0.09,
                economic_rationale="CONTRARIAN: Gasoline weakness during traditional driving season prep",
                volatility_adjustment=0.7
            ),
            
            # Crude Oil - CONTRARIAN: Short summer driving season
            SeasonalPosition(
                commodity='CL_F',
                direction=-1,  # SHORT during traditional driving season
                entry_month=5,   # May
                exit_month=8,    # August
                expected_return=0.05,
                statistical_significance=0.10,
                economic_rationale="CONTRARIAN: Crude oil weakness during traditional driving season",
                volatility_adjustment=0.5
            )
        ]
        
        return positions


class ContrarianAgriculturalStrategy(BaseSeasonalStrategy):
    """
    Contrarian agricultural strategy - opposite of traditional harvest cycles.
    
    Takes advantage of inverted agricultural patterns:
    - Long during traditional harvest pressure periods
    - Short during traditional post-harvest strength periods
    """
    
    def __init__(self):
        super().__init__(
            name="Contrarian Agricultural Strategy",
            description="Agricultural seasonality with inverted harvest cycle logic"
        )
    
    def define_seasonal_positions(self) -> List[SeasonalPosition]:
        """Define contrarian agricultural seasonal positions"""
        positions = [
            # Corn - CONTRARIAN: Long during harvest pressure
            SeasonalPosition(
                commodity='ZC_F',
                direction=1,   # LONG during traditional harvest pressure
                entry_month=7,   # July
                exit_month=9,    # September
                expected_return=0.08,
                statistical_significance=0.08,
                economic_rationale="CONTRARIAN: Corn strength during traditional harvest pressure",
                volatility_adjustment=0.8
            ),
            
            # Corn - CONTRARIAN: Short during post-harvest
            SeasonalPosition(
                commodity='ZC_F',
                direction=-1,  # SHORT during traditional post-harvest strength
                entry_month=11,  # November
                exit_month=1,    # January
                expected_return=0.06,
                statistical_significance=0.09,
                economic_rationale="CONTRARIAN: Corn weakness during traditional post-harvest period",
                volatility_adjustment=0.8
            ),
            
            # Wheat - CONTRARIAN: Short during planting season
            SeasonalPosition(
                commodity='ZW_F',
                direction=-1,  # SHORT during traditional planting strength
                entry_month=9,   # September
                exit_month=11,   # November
                expected_return=0.05,
                statistical_significance=0.10,
                economic_rationale="CONTRARIAN: Wheat weakness during traditional planting season",
                volatility_adjustment=0.9
            ),
            
            # Wheat - CONTRARIAN: Long during harvest season
            SeasonalPosition(
                commodity='ZW_F',
                direction=1,   # LONG during traditional harvest weakness
                entry_month=5,   # May
                exit_month=7,    # July
                expected_return=0.07,
                statistical_significance=0.08,
                economic_rationale="CONTRARIAN: Wheat strength during traditional harvest season",
                volatility_adjustment=0.9
            ),
            
            # Sugar - CONTRARIAN: Short during crush season
            SeasonalPosition(
                commodity='SB_F',
                direction=-1,  # SHORT during traditional crush season strength
                entry_month=9,   # September
                exit_month=11,   # November
                expected_return=0.04,
                statistical_significance=0.12,
                economic_rationale="CONTRARIAN: Sugar weakness during traditional crush season",
                volatility_adjustment=0.7
            ),
            
            # Sugar - CONTRARIAN: Long during off-season
            SeasonalPosition(
                commodity='SB_F',
                direction=1,   # LONG during traditional off-season weakness
                entry_month=2,   # February
                exit_month=4,    # April
                expected_return=0.06,
                statistical_significance=0.10,
                economic_rationale="CONTRARIAN: Sugar strength during traditional off-season",
                volatility_adjustment=0.7
            )
        ]
        
        return positions


class ContrarianMetalsStrategy(BaseSeasonalStrategy):
    """
    Contrarian metals strategy - opposite of traditional calendar effects.
    
    Inverts traditional metals patterns:
    - Short during January Effect period
    - Long during May weakness period
    - Opposite industrial demand cycles
    """
    
    def __init__(self):
        super().__init__(
            name="Contrarian Metals Strategy",
            description="Metals seasonality with inverted calendar effects"
        )
    
    def define_seasonal_positions(self) -> List[SeasonalPosition]:
        """Define contrarian metals seasonal positions"""
        positions = [
            # Gold - CONTRARIAN: Short January Effect
            SeasonalPosition(
                commodity='GC_F',
                direction=-1,  # SHORT during traditional January Effect
                entry_month=12,  # December
                exit_month=1,    # January
                expected_return=0.03,
                statistical_significance=0.12,
                economic_rationale="CONTRARIAN: Gold weakness during traditional January Effect",
                volatility_adjustment=1.2
            ),
            
            # Gold - CONTRARIAN: Long May weakness
            SeasonalPosition(
                commodity='GC_F',
                direction=1,   # LONG during traditional May weakness
                entry_month=4,   # April
                exit_month=5,    # May
                expected_return=0.04,
                statistical_significance=0.11,
                economic_rationale="CONTRARIAN: Gold strength during traditional May weakness",
                volatility_adjustment=1.2
            ),
            
            # Silver - CONTRARIAN: Short January Effect
            SeasonalPosition(
                commodity='SI_F',
                direction=-1,  # SHORT during traditional January Effect
                entry_month=12,  # December
                exit_month=1,    # January
                expected_return=0.05,
                statistical_significance=0.10,
                economic_rationale="CONTRARIAN: Silver weakness during traditional January Effect",
                volatility_adjustment=1.0
            ),
            
            # Silver - CONTRARIAN: Long May weakness
            SeasonalPosition(
                commodity='SI_F',
                direction=1,   # LONG during traditional May weakness
                entry_month=4,   # April
                exit_month=5,    # May
                expected_return=0.06,
                statistical_significance=0.09,
                economic_rationale="CONTRARIAN: Silver strength during traditional May weakness",
                volatility_adjustment=1.0
            ),
            
            # Copper - CONTRARIAN: Short Chinese New Year restart
            SeasonalPosition(
                commodity='HG_F',
                direction=-1,  # SHORT during traditional industrial restart
                entry_month=1,   # January
                exit_month=3,    # March
                expected_return=0.04,
                statistical_significance=0.11,
                economic_rationale="CONTRARIAN: Copper weakness during traditional industrial restart",
                volatility_adjustment=0.9
            ),
            
            # Copper - CONTRARIAN: Long summer industrial slowdown
            SeasonalPosition(
                commodity='HG_F',
                direction=1,   # LONG during traditional industrial slowdown
                entry_month=7,   # July
                exit_month=9,    # September
                expected_return=0.05,
                statistical_significance=0.10,
                economic_rationale="CONTRARIAN: Copper strength during traditional industrial slowdown",
                volatility_adjustment=0.9
            )
        ]
        
        return positions


class AdaptiveSeasonalStrategy(BaseSeasonalStrategy):
    """
    Adaptive seasonal strategy that automatically chooses between normal and contrarian logic
    based on recent performance and regime detection.
    """
    
    def __init__(self):
        super().__init__(
            name="Adaptive Seasonal Strategy",
            description="Automatically adapts between normal and contrarian seasonal logic"
        )
        
        # Initialize component strategies
        self.normal_energy = None  # Will be imported if needed
        self.normal_agricultural = None
        self.normal_metals = None
        
        self.contrarian_energy = ContrarianEnergyStrategy()
        self.contrarian_agricultural = ContrarianAgriculturalStrategy()
        self.contrarian_metals = ContrarianMetalsStrategy()
        
        self.lookback_period = 252 * 2  # 2 years for regime detection
    
    def define_seasonal_positions(self) -> List[SeasonalPosition]:
        """Combine all contrarian positions for now"""
        all_positions = []
        all_positions.extend(self.contrarian_energy.define_seasonal_positions())
        all_positions.extend(self.contrarian_agricultural.define_seasonal_positions())
        all_positions.extend(self.contrarian_metals.define_seasonal_positions())
        return all_positions
    
    def detect_regime(self, returns_data: pd.DataFrame, 
                     seasonal_stats: Dict, commodity: str) -> str:
        """
        Detect whether to use normal or contrarian logic for a commodity.
        
        Returns:
            'normal' or 'contrarian' based on recent pattern performance
        """
        if len(returns_data) < self.lookback_period:
            return 'contrarian'  # Default to contrarian given our analysis
        
        # Use recent data for regime detection
        recent_data = returns_data.iloc[-self.lookback_period:]
        
        # Analyze recent patterns vs expectations
        analyzer = PatternInversionAnalyzer()
        recent_inversions = analyzer.analyze_pattern_inversions(recent_data)
        
        if commodity in recent_inversions:
            is_inverted = recent_inversions[commodity]['is_inverted']
            return 'contrarian' if is_inverted else 'normal'
        
        return 'contrarian'  # Default given our overall findings
    
    def generate_adaptive_signals(self, returns_data: pd.DataFrame,
                                seasonal_stats: Dict) -> pd.DataFrame:
        """Generate signals using adaptive regime detection"""
        
        # For now, use contrarian logic as our analysis shows inversions
        signals = self.generate_signals(returns_data, seasonal_stats)
        
        # Future enhancement: implement dynamic switching based on regime detection
        # For each commodity, detect regime and switch strategy accordingly
        
        return signals


class ContrarianSectorRotationStrategy(BaseSeasonalStrategy):
    """
    Contrarian sector rotation strategy that inverts traditional sector timing.
    """
    
    def __init__(self):
        super().__init__(
            name="Contrarian Sector Rotation Strategy",
            description="Multi-sector contrarian rotation with inverted seasonal timing"
        )
        
        # Initialize component strategies
        self.contrarian_energy = ContrarianEnergyStrategy()
        self.contrarian_agricultural = ContrarianAgriculturalStrategy()
        self.contrarian_metals = ContrarianMetalsStrategy()
    
    def define_seasonal_positions(self) -> List[SeasonalPosition]:
        """Combine all contrarian sector positions"""
        all_positions = []
        all_positions.extend(self.contrarian_energy.define_seasonal_positions())
        all_positions.extend(self.contrarian_agricultural.define_seasonal_positions())
        all_positions.extend(self.contrarian_metals.define_seasonal_positions())
        return all_positions
    
    def generate_sector_allocation(self, returns_data: pd.DataFrame,
                                 seasonal_stats: Dict) -> pd.DataFrame:
        """
        Generate contrarian sector allocation weights.
        
        Allocates more weight to sectors during their traditionally weak periods.
        """
        sector_mapping = {
            'Energy': ['CL_F', 'NG_F', 'RB_F', 'HO_F', 'BZ_F'],
            'Metals': ['GC_F', 'SI_F', 'HG_F', 'PA_F'],
            'Agriculture': ['ZC_F', 'ZW_F', 'ZS_F', 'SB_F', 'CT_F', 'CC_F']
        }
        
        # Initialize allocation DataFrame
        allocations = pd.DataFrame(
            index=returns_data.index,
            columns=sector_mapping.keys()
        )
        
        if 'sector_seasonality' not in seasonal_stats:
            # Equal weight if no seasonal stats
            allocations[:] = 1/3
            return allocations
        
        sector_seasonality = seasonal_stats['sector_seasonality']
        
        for date in returns_data.index:
            month_name = date.strftime('%b')
            if month_name in sector_seasonality.index:
                monthly_returns = sector_seasonality.loc[month_name]
                
                # CONTRARIAN allocation: allocate more to weakest performing sectors
                inverted_returns = -monthly_returns  # Invert the returns
                exp_returns = np.exp(inverted_returns * 10)  # Scale for allocation
                weights = exp_returns / exp_returns.sum()
                allocations.loc[date] = weights
            else:
                allocations.loc[date] = 1/3  # Equal weight fallback
        
        return allocations


def main():
    """Test contrarian seasonal strategies"""
    print("=== CONTRARIAN SEASONAL STRATEGIES TEST ===\n")
    
    # Initialize strategies
    strategies = {
        'Contrarian_Energy': ContrarianEnergyStrategy(),
        'Contrarian_Agricultural': ContrarianAgriculturalStrategy(),
        'Contrarian_Metals': ContrarianMetalsStrategy(),
        'Contrarian_Sector_Rotation': ContrarianSectorRotationStrategy(),
        'Adaptive_Seasonal': AdaptiveSeasonalStrategy()
    }
    
    # Display strategy definitions
    for name, strategy in strategies.items():
        print(f"📊 {name.replace('_', ' ').upper()}")
        print("-" * 50)
        
        positions = strategy.define_seasonal_positions()
        
        for i, pos in enumerate(positions[:3], 1):  # Show first 3 positions
            direction = "SHORT" if pos.direction == -1 else "LONG"
            
            if pos.entry_month > pos.exit_month:
                period = f"Month {pos.entry_month}-12 & 1-{pos.exit_month}"
            else:
                period = f"Month {pos.entry_month}-{pos.exit_month}"
            
            print(f"  {i}. {pos.commodity} - {direction}")
            print(f"     Period: {period}")
            print(f"     Expected Return: {pos.expected_return:.1%}")
            print(f"     Rationale: {pos.economic_rationale}")
            print()
        
        if len(positions) > 3:
            print(f"  ... and {len(positions) - 3} more positions")
        print()
    
    print("✅ Contrarian seasonal strategies initialized!")
    print("These strategies invert traditional seasonal logic based on pattern analysis.")


if __name__ == "__main__":
    main()