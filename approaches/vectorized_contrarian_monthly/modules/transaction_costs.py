import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class IBKRCommissionTier:
    """Data class for IBKR commission tier structure"""
    min_volume: int
    max_volume: Optional[int]
    execution_fee: float
    description: str


class IBKRFeeCalculator:
    """
    Interactive Brokers futures commission calculator
    Based on IBKR tiered pricing structure as of 2024-2025
    """
    
    # IBKR Volume-based tiered pricing (monthly cumulative)
    COMMISSION_TIERS = [
        IBKRCommissionTier(0, 1000, 0.85, "Tier 1: 0-1,000 contracts"),
        IBKRCommissionTier(1001, 10000, 0.65, "Tier 2: 1,001-10,000 contracts"),
        IBKRCommissionTier(10001, 20000, 0.45, "Tier 3: 10,001-20,000 contracts"),
        IBKRCommissionTier(20001, None, 0.25, "Tier 4: 20,001+ contracts")
    ]
    
    # Additional fee components (approximate averages across commodity futures)
    EXCHANGE_FEE = 1.38  # Average exchange fee per contract
    CLEARING_FEE = 0.00  # Usually minimal or zero
    REGULATORY_FEE = 0.02  # Small regulatory fee per contract
    
    def __init__(self):
        self.monthly_volume_tracker = {}
        self.total_costs_tracker = {}
    
    def reset_monthly_tracking(self, month_key: str) -> None:
        """Reset monthly volume tracking for a new month"""
        self.monthly_volume_tracker[month_key] = 0
        self.total_costs_tracker[month_key] = 0.0
    
    def calculate_marginal_commission(self, contracts: int, current_monthly_volume: int = 0) -> Tuple[float, Dict]:
        """
        Calculate commission using marginal tier pricing
        
        Args:
            contracts: Number of contracts to trade
            current_monthly_volume: Current monthly volume before this trade
            
        Returns:
            Tuple of (total_commission, breakdown_dict)
        """
        if contracts <= 0:
            return 0.0, {}
        
        remaining_contracts = contracts
        total_execution_fee = 0.0
        tier_breakdown = {}
        
        current_volume = current_monthly_volume
        
        for tier in self.COMMISSION_TIERS:
            if remaining_contracts <= 0:
                break
                
            # Calculate how many contracts fall in this tier
            tier_start = max(tier.min_volume, current_volume + 1) if current_volume >= tier.min_volume else tier.min_volume
            tier_end = tier.max_volume if tier.max_volume else float('inf')
            
            # Skip if we're already past this tier
            if current_volume >= tier_end:
                continue
            
            # Calculate contracts in this tier
            contracts_in_tier = min(remaining_contracts, tier_end - current_volume)
            contracts_in_tier = max(0, contracts_in_tier)
            
            if contracts_in_tier > 0:
                tier_cost = contracts_in_tier * tier.execution_fee
                total_execution_fee += tier_cost
                tier_breakdown[tier.description] = {
                    'contracts': contracts_in_tier,
                    'rate': tier.execution_fee,
                    'cost': tier_cost
                }
                
                remaining_contracts -= contracts_in_tier
                current_volume += contracts_in_tier
        
        # Calculate total commission including all fee components
        total_other_fees = contracts * (self.EXCHANGE_FEE + self.CLEARING_FEE + self.REGULATORY_FEE)
        total_commission = total_execution_fee + total_other_fees
        
        breakdown = {
            'contracts': contracts,
            'execution_fee': total_execution_fee,
            'exchange_fee': contracts * self.EXCHANGE_FEE,
            'clearing_fee': contracts * self.CLEARING_FEE,
            'regulatory_fee': contracts * self.REGULATORY_FEE,
            'total_commission': total_commission,
            'avg_per_contract': total_commission / contracts if contracts > 0 else 0,
            'tier_breakdown': tier_breakdown
        }
        
        return total_commission, breakdown
    
    def get_fee_summary(self) -> Dict:
        """Get summary of fee structure"""
        return {
            'base_fees': {
                'exchange_fee': self.EXCHANGE_FEE,
                'clearing_fee': self.CLEARING_FEE,
                'regulatory_fee': self.REGULATORY_FEE,
                'total_base_fees': self.EXCHANGE_FEE + self.CLEARING_FEE + self.REGULATORY_FEE
            },
            'execution_fee_tiers': [
                {
                    'tier': i+1,
                    'volume_range': f"{tier.min_volume:,}-{tier.max_volume:,}" if tier.max_volume else f"{tier.min_volume:,}+",
                    'execution_fee': tier.execution_fee,
                    'total_per_contract': tier.execution_fee + self.EXCHANGE_FEE + self.CLEARING_FEE + self.REGULATORY_FEE
                }
                for i, tier in enumerate(self.COMMISSION_TIERS)
            ]
        }


def detect_drawdown_filter_transactions(investment_state: pd.Series, 
                                       positions_df: pd.DataFrame,
                                       contract_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Detect transactions caused specifically by drawdown filter activation/deactivation
    
    Args:
        investment_state: Series indicating when strategy is invested (1) or not (0)
        positions_df: DataFrame with commodity positions over time
        contract_multiplier: Multiplier to convert position weights to contract counts
        
    Returns:
        DataFrame with drawdown filter transaction details
    """
    dd_transactions = []
    
    # Find all state changes in investment
    state_changes = investment_state.diff()
    
    for date in state_changes.index:
        if pd.isna(state_changes.loc[date]):
            continue
            
        change = state_changes.loc[date]
        current_state = investment_state.loc[date]
        
        # Find the most recent position data for this date
        available_pos_dates = positions_df.index[positions_df.index <= date]
        if len(available_pos_dates) == 0:
            continue
            
        position_date = available_pos_dates[-1]
        positions = positions_df.loc[position_date]
        total_contracts = positions.abs().sum() * contract_multiplier
        
        if abs(change) > 0 and total_contracts > 0:
            transaction_type = ""
            description = ""
            
            if change == -1:  # Going from invested (1) to not invested (0)
                transaction_type = "DD_EXIT"
                description = f"Drawdown filter exit: close all {total_contracts:.1f} contracts"
            elif change == 1:  # Going from not invested (0) to invested (1)  
                transaction_type = "DD_ENTRY"
                description = f"Drawdown filter entry: open all {total_contracts:.1f} contracts"
            
            if transaction_type:
                dd_transactions.append({
                    'date': date,
                    'transaction_type': transaction_type,
                    'contracts': total_contracts,
                    'description': description,
                    'positions_breakdown': positions.to_dict()
                })
    
    df = pd.DataFrame(dd_transactions)
    return df.set_index('date') if not df.empty else pd.DataFrame()


def detect_position_changes(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect position changes that trigger transaction costs
    
    Args:
        positions_df: DataFrame with commodity positions over time
        
    Returns:
        DataFrame with transaction details (entries, exits, position changes)
    """
    # Calculate position changes
    position_changes = positions_df.diff().fillna(positions_df.iloc[0])
    
    # Calculate absolute position changes (total contracts traded)
    abs_changes = position_changes.abs()
    
    # Create transaction summary
    transactions = []
    
    for date in position_changes.index:
        monthly_transactions = {}
        total_contracts = 0
        
        for commodity in position_changes.columns:
            change = position_changes.loc[date, commodity]
            abs_change = abs(change)
            
            if abs_change > 1e-6:  # Avoid tiny floating point changes
                monthly_transactions[commodity] = {
                    'position_change': change,
                    'contracts_traded': abs_change,
                    'trade_type': 'entry' if positions_df.loc[date, commodity] != 0 and change != 0 else 'exit'
                }
                total_contracts += abs_change
        
        transactions.append({
            'date': date,
            'total_contracts': total_contracts,
            'commodity_transactions': monthly_transactions,
            'n_commodities_traded': len(monthly_transactions)
        })
    
    return pd.DataFrame(transactions).set_index('date')


def apply_transaction_costs_to_equity(equity_df: pd.DataFrame,
                                    positions_df: pd.DataFrame,
                                    investment_state: pd.Series = None,
                                    contract_multiplier: float = 1.0,
                                    cost_basis_method: str = 'percentage') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply IBKR transaction costs to equity curve considering monthly rebalancing and filter exits/entries
    
    Args:
        equity_df: DataFrame with cumulative returns
        positions_df: DataFrame with commodity positions over time
        investment_state: Series indicating when strategy is invested (from drawdown filter)
        contract_multiplier: Multiplier to convert position weights to contract counts
        cost_basis_method: 'percentage' (costs as % of equity) or 'absolute' (fixed dollar costs)
        
    Returns:
        Tuple of (equity_with_costs, transaction_costs_breakdown)
    """
    fee_calculator = IBKRFeeCalculator()
    
    # Initialize result DataFrames
    equity_with_costs = equity_df.copy()
    transaction_details = []
    
    # Check for date alignment between equity and positions
    equity_start = equity_df.index.min()
    equity_end = equity_df.index.max()
    positions_start = positions_df.index.min()
    positions_end = positions_df.index.max()
    
    print(f"Date alignment check:")
    print(f"  Equity: {equity_start.date()} to {equity_end.date()}")
    print(f"  Positions: {positions_start.date()} to {positions_end.date()}")
    
    # Find overlapping date range
    overlap_start = max(equity_start, positions_start)
    overlap_end = min(equity_end, positions_end)
    
    if overlap_start > overlap_end:
        print("Warning: No date overlap between equity and positions data")
        return equity_with_costs, pd.DataFrame()
    
    # Filter positions to overlapping period and align with equity data frequency
    positions_aligned = positions_df.loc[overlap_start:overlap_end]
    
    # If daily equity data, we need to forward-fill monthly positions
    if len(equity_df) > len(positions_df) * 20:  # Heuristic for daily vs monthly
        print("  Aligning monthly positions to daily equity data...")
        # Reindex positions to daily frequency, forward filling within each month
        positions_aligned = positions_aligned.reindex(
            equity_df.loc[overlap_start:overlap_end].index, 
            method='ffill'
        ).fillna(0)
    
    # Detect all position changes (monthly rebalancing)
    transactions_df = detect_position_changes(positions_aligned)
    
    if transactions_df.empty or transactions_df['total_contracts'].sum() == 0:
        print("Warning: No meaningful position changes detected for transaction cost calculation")
        return equity_with_costs, pd.DataFrame()
    
    # Track cumulative costs
    cumulative_costs = 0.0
    monthly_volume = 0
    current_month_key = None
    
    for date in transactions_df.index:
        # Get month key for volume tracking
        month_key = date.strftime('%Y-%m')
        
        # Reset monthly tracking if new month
        if month_key != current_month_key:
            fee_calculator.reset_monthly_tracking(month_key)
            monthly_volume = 0
            current_month_key = month_key
        
        # Get transaction info
        total_contracts = transactions_df.loc[date, 'total_contracts'] * contract_multiplier
        
        # Skip if no meaningful trading
        if total_contracts < 0.01:
            continue
        
        # Handle drawdown filter state changes
        additional_costs = 0.0
        if investment_state is not None:
            # Find closest date in investment_state
            try:
                # Try exact date match first
                if date in investment_state.index:
                    current_state = investment_state.loc[date]
                    prev_state = investment_state.shift(1).loc[date] if date in investment_state.shift(1).index else 1
                else:
                    # Find nearest date (for monthly positions on daily investment state)
                    nearest_date = investment_state.index[investment_state.index <= date]
                    if len(nearest_date) > 0:
                        nearest_date = nearest_date[-1]  # Latest date before or on transaction date
                        current_state = investment_state.loc[nearest_date]
                        prev_state = investment_state.shift(1).loc[nearest_date] if nearest_date in investment_state.shift(1).index else 1
                    else:
                        current_state = 1
                        prev_state = 1
            except Exception as e:
                print(f"Warning: Could not align investment state for date {date}: {e}")
                current_state = 1
                prev_state = 1
            
            # If exiting due to drawdown filter, add exit costs for all positions
            if prev_state == 1 and current_state == 0:
                # Calculate exit costs for all current positions
                try:
                    current_positions = positions_aligned.loc[date].abs().sum() * contract_multiplier
                except KeyError:
                    # Use the most recent available position data
                    available_dates = positions_aligned.index[positions_aligned.index <= date]
                    if len(available_dates) > 0:
                        current_positions = positions_aligned.loc[available_dates[-1]].abs().sum() * contract_multiplier
                    else:
                        current_positions = 0
                if current_positions > 0:
                    exit_cost, exit_breakdown = fee_calculator.calculate_marginal_commission(
                        int(current_positions), monthly_volume
                    )
                    additional_costs += exit_cost
                    monthly_volume += int(current_positions)
                    
            # If re-entering due to drawdown filter recovery, add entry costs
            elif prev_state == 0 and current_state == 1:
                # Calculate entry costs for new positions
                try:
                    new_positions = positions_aligned.loc[date].abs().sum() * contract_multiplier
                except KeyError:
                    # Use the most recent available position data
                    available_dates = positions_aligned.index[positions_aligned.index <= date]
                    if len(available_dates) > 0:
                        new_positions = positions_aligned.loc[available_dates[-1]].abs().sum() * contract_multiplier
                    else:
                        new_positions = 0
                if new_positions > 0:
                    entry_cost, entry_breakdown = fee_calculator.calculate_marginal_commission(
                        int(new_positions), monthly_volume
                    )
                    additional_costs += entry_cost
                    monthly_volume += int(new_positions)
        
        # Calculate regular rebalancing costs
        if total_contracts > 0:
            commission, breakdown = fee_calculator.calculate_marginal_commission(
                int(total_contracts), monthly_volume
            )
            monthly_volume += int(total_contracts)
        else:
            commission, breakdown = 0.0, {}
        
        # Total costs for this period
        total_period_costs = commission + additional_costs
        cumulative_costs += total_period_costs
        
        # Apply costs to equity curve
        if cost_basis_method == 'percentage':
            # Apply as percentage of current equity value
            current_equity = equity_with_costs.loc[date, 'cumulative_returns']
            cost_impact = total_period_costs / (current_equity * 100000)  # Assume $100k initial capital
            equity_with_costs.loc[date:, 'cumulative_returns'] *= (1 - cost_impact)
        else:
            # Apply as absolute dollar cost (would need portfolio value)
            pass  # Implementation depends on absolute portfolio size
        
        # Store transaction details
        transaction_details.append({
            'date': date,
            'regular_commission': commission,
            'dd_filter_costs': additional_costs,
            'total_costs': total_period_costs,
            'cumulative_costs': cumulative_costs,
            'contracts_traded': total_contracts,
            'dd_filter_volume': int(additional_costs / 2.25) if additional_costs > 0 else 0,  # Estimate contracts from costs
            'regular_volume': int(total_contracts),
            'monthly_volume': monthly_volume,
            'commission_breakdown': breakdown
        })
    
    # Create transaction costs DataFrame
    costs_df = pd.DataFrame(transaction_details)
    if not costs_df.empty:
        costs_df = costs_df.set_index('date')
    
    # Add cost-adjusted returns column
    if not costs_df.empty:
        equity_with_costs['cumulative_returns_with_costs'] = equity_with_costs['cumulative_returns']
        
        # Ensure we have the costs data aligned with equity data
        for date in costs_df.index:
            if date in equity_with_costs.index:
                cost_impact = costs_df.loc[date, 'total_costs']
                if cost_basis_method == 'percentage':
                    # Small percentage impact
                    impact_ratio = 1 - (cost_impact / 100000)  # Assuming $100k base
                    equity_with_costs.loc[date:, 'cumulative_returns_with_costs'] *= impact_ratio
    else:
        equity_with_costs['cumulative_returns_with_costs'] = equity_with_costs['cumulative_returns']
    
    return equity_with_costs, costs_df


def analyze_cost_impact(original_equity: pd.Series,
                       equity_with_costs: pd.Series,
                       transaction_costs_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the impact of transaction costs on strategy performance
    
    Args:
        original_equity: Original equity curve without costs
        equity_with_costs: Equity curve with transaction costs applied
        transaction_costs_df: DataFrame with transaction cost details
        
    Returns:
        Dictionary with cost impact analysis
    """
    # Basic performance comparison
    original_final = original_equity.iloc[-1]
    costs_final = equity_with_costs.iloc[-1]
    
    # Calculate cost impact metrics
    total_return_impact = (costs_final / original_final) - 1
    
    # Annualized cost impact
    years = len(original_equity) / 252  # Assuming daily data
    annual_cost_impact = ((costs_final / original_final) ** (1/years)) - 1 if years > 0 else 0
    
    # Total costs summary
    if not transaction_costs_df.empty:
        total_costs_dollars = transaction_costs_df['total_costs'].sum()
        total_contracts_traded = transaction_costs_df['contracts_traded'].sum()
        avg_cost_per_contract = total_costs_dollars / total_contracts_traded if total_contracts_traded > 0 else 0
        
        # Monthly cost statistics
        monthly_costs = transaction_costs_df.groupby(transaction_costs_df.index.to_period('M'))['total_costs'].sum()
        avg_monthly_costs = monthly_costs.mean()
        max_monthly_costs = monthly_costs.max()
    else:
        total_costs_dollars = 0
        total_contracts_traded = 0
        avg_cost_per_contract = 0
        avg_monthly_costs = 0
        max_monthly_costs = 0
    
    # Calculate Sharpe ratio impact
    orig_returns = original_equity.pct_change().dropna()
    cost_returns = equity_with_costs.pct_change().dropna()
    
    orig_sharpe = (orig_returns.mean() * 252) / (orig_returns.std() * np.sqrt(252)) if len(orig_returns) > 0 else 0
    cost_sharpe = (cost_returns.mean() * 252) / (cost_returns.std() * np.sqrt(252)) if len(cost_returns) > 0 else 0
    
    return {
        'total_return_impact_pct': total_return_impact * 100,
        'annual_cost_impact_pct': annual_cost_impact * 100,
        'total_costs_dollars': total_costs_dollars,
        'total_contracts_traded': total_contracts_traded,
        'avg_cost_per_contract': avg_cost_per_contract,
        'avg_monthly_costs': avg_monthly_costs,
        'max_monthly_costs': max_monthly_costs,
        'original_sharpe': orig_sharpe,
        'costs_sharpe': cost_sharpe,
        'sharpe_impact': cost_sharpe - orig_sharpe,
        'cost_as_pct_of_returns': abs(total_return_impact) / (original_final - 1) * 100 if original_final > 1 else 0
    }


def plot_transaction_costs_analysis(equity_df: pd.DataFrame,
                                  costs_df: pd.DataFrame,
                                  cost_impact: Dict[str, float],
                                  dd_transactions_df: pd.DataFrame = None,
                                  figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Create comprehensive visualization of transaction costs impact
    
    Args:
        equity_df: DataFrame with original and cost-adjusted equity curves
        costs_df: DataFrame with transaction cost details
        cost_impact: Cost impact analysis from analyze_cost_impact
        figsize: Figure size for plots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1])
    
    # Main equity curve comparison
    ax1 = fig.add_subplot(gs[0, :])
    if 'cumulative_returns' in equity_df.columns:
        ax1.plot(equity_df.index, equity_df['cumulative_returns'], 
                label='Without Transaction Costs', linewidth=2, alpha=0.8)
    if 'cumulative_returns_with_costs' in equity_df.columns:
        ax1.plot(equity_df.index, equity_df['cumulative_returns_with_costs'], 
                label='With IBKR Transaction Costs', linewidth=2, alpha=0.8)
    
    ax1.set_title('Strategy Performance: Impact of Transaction Costs')
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    if not costs_df.empty:
        # Monthly transaction costs - separate regular vs DD filter costs
        ax2 = fig.add_subplot(gs[1, :])
        monthly_regular = costs_df.groupby(costs_df.index.to_period('M'))['regular_commission'].sum()
        monthly_dd = costs_df.groupby(costs_df.index.to_period('M'))['dd_filter_costs'].sum()
        
        # Stacked bar chart
        ax2.bar(range(len(monthly_regular)), monthly_regular, alpha=0.7, label='Regular Rebalancing', color='steelblue')
        ax2.bar(range(len(monthly_dd)), monthly_dd, bottom=monthly_regular, alpha=0.7, label='DD Filter Events', color='orange')
        
        ax2.set_title('Monthly Transaction Costs Breakdown')
        ax2.set_ylabel('Cost ($)')
        ax2.set_xlabel('Month')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Set x-axis labels to show actual months
        if len(monthly_regular) > 20:
            step = max(1, len(monthly_regular) // 10)
            tick_positions = range(0, len(monthly_regular), step)
            tick_labels = [str(monthly_regular.index[i]) for i in tick_positions]
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)
        else:
            ax2.set_xticks(range(len(monthly_regular)))
            ax2.set_xticklabels([str(x) for x in monthly_regular.index])
        
        # Cumulative costs
        ax3 = fig.add_subplot(gs[2, 0])
        costs_df['cumulative_costs'].plot(ax=ax3, color='red', linewidth=2)
        ax3.set_title('Cumulative Transaction Costs')
        ax3.set_ylabel('Total Costs ($)')
        ax3.grid(True, alpha=0.3)
        
        # Contracts traded over time
        ax4 = fig.add_subplot(gs[2, 1])
        costs_df['contracts_traded'].plot(ax=ax4, color='orange', alpha=0.7)
        ax4.set_title('Contracts Traded per Month')
        ax4.set_ylabel('Contracts')
        ax4.grid(True, alpha=0.3)
        
        # Cost per contract over time
        ax5 = fig.add_subplot(gs[2, 2])
        cost_per_contract = costs_df['total_costs'] / costs_df['contracts_traded']
        cost_per_contract.plot(ax=ax5, color='green', alpha=0.7)
        ax5.set_title('Cost per Contract')
        ax5.set_ylabel('$/Contract')
        ax5.grid(True, alpha=0.3)
    
    # Cost impact summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate DD vs regular cost breakdown if available
    dd_costs_total = costs_df['dd_filter_costs'].sum() if 'dd_filter_costs' in costs_df.columns else 0
    regular_costs_total = costs_df['regular_commission'].sum() if 'regular_commission' in costs_df.columns else cost_impact['total_costs_dollars']
    dd_percentage = (dd_costs_total / cost_impact['total_costs_dollars'] * 100) if cost_impact['total_costs_dollars'] > 0 else 0
    
    summary_text = f"""Transaction Costs Impact Summary (IBKR Fee Structure):

Performance Impact:
• Total Return Impact: {cost_impact['total_return_impact_pct']:.2f}%
• Annual Cost Impact: {cost_impact['annual_cost_impact_pct']:.2f}%
• Sharpe Ratio Impact: {cost_impact['sharpe_impact']:.3f}

Cost Breakdown:
• Total Costs: ${cost_impact['total_costs_dollars']:.2f}
• Regular Rebalancing: ${regular_costs_total:.2f} ({100-dd_percentage:.1f}%)
• DD Filter Events: ${dd_costs_total:.2f} ({dd_percentage:.1f}%)
• Avg Cost/Contract: ${cost_impact['avg_cost_per_contract']:.2f}

Trading Statistics:
• Total Contracts: {cost_impact['total_contracts_traded']:.0f}
• Avg Monthly Costs: ${cost_impact['avg_monthly_costs']:.2f}
• Max Monthly Costs: ${cost_impact['max_monthly_costs']:.2f}

Cost Efficiency:
• Costs as % of Returns: {cost_impact['cost_as_pct_of_returns']:.2f}%
• Original Sharpe: {cost_impact['original_sharpe']:.3f}
• With Costs Sharpe: {cost_impact['costs_sharpe']:.3f}"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             verticalalignment='top', fontsize=11, fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()


def create_cost_sensitivity_analysis(positions_df: pd.DataFrame,
                                   base_equity: pd.Series,
                                   contract_multipliers: List[float] = [0.5, 1.0, 2.0, 5.0],
                                   investment_state: pd.Series = None) -> pd.DataFrame:
    """
    Analyze sensitivity to different contract sizing assumptions
    
    Args:
        positions_df: Position weights DataFrame
        base_equity: Base equity curve without costs
        contract_multipliers: Different multipliers to test
        investment_state: Investment state from drawdown filter
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    sensitivity_results = []
    
    for multiplier in contract_multipliers:
        # Apply transaction costs with this multiplier
        equity_with_costs, costs_df = apply_transaction_costs_to_equity(
            base_equity.to_frame('cumulative_returns'),
            positions_df,
            investment_state,
            contract_multiplier=multiplier
        )
        
        # Analyze impact
        if 'cumulative_returns_with_costs' in equity_with_costs.columns:
            cost_impact = analyze_cost_impact(
                base_equity,
                equity_with_costs['cumulative_returns_with_costs'],
                costs_df
            )
        else:
            cost_impact = analyze_cost_impact(base_equity, base_equity, pd.DataFrame())
        
        sensitivity_results.append({
            'contract_multiplier': multiplier,
            'assumed_portfolio_size': f"${multiplier * 100000:,.0f}",
            **cost_impact
        })
    
    return pd.DataFrame(sensitivity_results)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing IBKR transaction costs module...")
    
    # Test fee calculator
    fee_calc = IBKRFeeCalculator()
    
    print("\nIBKR Fee Structure:")
    fee_summary = fee_calc.get_fee_summary()
    print(f"Base fees per contract: ${fee_summary['base_fees']['total_base_fees']:.2f}")
    
    for tier in fee_summary['execution_fee_tiers']:
        print(f"Tier {tier['tier']}: {tier['volume_range']} contracts = ${tier['total_per_contract']:.2f}/contract")
    
    # Test commission calculation
    print(f"\nTesting commission calculations:")
    test_volumes = [100, 1500, 15000, 25000]
    monthly_volume = 0
    
    for volume in test_volumes:
        commission, breakdown = fee_calc.calculate_marginal_commission(volume, monthly_volume)
        print(f"{volume:,} contracts: ${commission:.2f} total (${breakdown['avg_per_contract']:.2f}/contract)")
        monthly_volume += volume
    
    print("\nTransaction costs module test completed successfully!")