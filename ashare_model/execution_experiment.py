"""P60 reporting only: observe an explicit historical replay; never execute it."""
from dataclasses import dataclass, replace

import numpy as np

from ashare_execution import ExecutionCostModel


@dataclass(frozen=True)
class StageDecision:
    admitted: bool
    reasons: tuple[str, ...]


def frequency_admission(active, passive):
    """Necessary development conditions, never a promotion/alpha judgment."""
    reasons = []
    if not active['valuation_complete'] or not passive['valuation_complete']:
        reasons.append('incomplete_valuation')
    if active['cash_min'] < 0:
        reasons.append('negative_cash')
    if active['first_funded_holdings'] < 20:
        reasons.append('initial_diversification')
    if active['fraction_funded_days_below_20'] > .05:
        reasons.append('persistent_diversification')
    if active['net_profit_rmb'] <= 0:
        reasons.append('no_positive_net_profit')
    if active['net_profit_rmb'] <= passive['net_profit_rmb']:
        reasons.append('no_positive_passive_increment')
    return StageDecision(not reasons, tuple(reasons))


def observe_legacy(result, config, raw, codes):
    """Recover reporting facts from native actual fills, without altering wealth.

    P35 daily marks with delay=1 only. Legacy rejections are unavailable;
    callers must explicitly label them, not interpret the empty adapter trail.
    """
    n = len(result.positions)
    if (not n or np.asarray(raw['open']).shape != (len(codes), n+2)
        or any(getattr(result, key) is None or len(getattr(result, key)) != n
            for key in ('buy_weights', 'sell_weights', 'cost_fractions', 'construction_diagnostics'))):
        raise ValueError('complete native daily execution trail required')
    if result.trades or any('account' in d for d in result.construction_diagnostics):
        raise ValueError('explicit unobserved historical result required')
    cost_model = ExecutionCostModel.from_config(config)
    cash = float(config.initial_capital)
    trades, diagnostics = [], []
    for i, position in enumerate(result.positions):
        before = float(result.equity_curve[i] * config.initial_capital)
        fee_sum, turnover = 0., 0.
        for side, weights in (('sell', result.sell_weights[i]), ('buy', result.buy_weights[i])):
            amounts = np.asarray(weights, dtype=float) * before
            if amounts.shape != (len(codes),) or np.any(~np.isfinite(amounts)) or np.any(amounts < 0):
                raise ValueError('invalid actual fills')
            for j in np.flatnonzero(amounts > 0):
                amount, price = float(amounts[j]), float(raw['open'][j, i+1])
                if not np.isfinite(price) or price <= 0:
                    raise ValueError('actual fill lacks nominal execution price')
                cost = cost_model.trade_cost(side, amount)
                fees = {name: float(getattr(cost, name)) for name in
                    ('commission', 'stamp_tax', 'transfer_fee', 'slippage', 'total')}
                cash += (amount if side == 'sell' else -amount) - fees['total']
                fee_sum += fees['total']; turnover += amount/before
                trades.append(dict(date=position['entry_date'], signal_date=position['signal_date'],
                    ts_code=codes[j], side=side, quantity=amount/price, price=price, amount=amount, **fees))
        if not np.isclose(fee_sum, result.cost_fractions[i]*before, rtol=1e-12, atol=1e-8):
            raise ValueError('native fee trail does not reconcile')
        if cash < 0:
            raise ValueError('negative historical cash')
        after = before-fee_sum
        if after <= 0:
            raise ValueError('nonpositive wealth')
        diagnostics.append(dict(**result.construction_diagnostics[i], execution_rejections=[],
            market_block_counts={}, account=dict(equity_before=before,
                equity_after_execution=after, equity_exit=float(result.equity_curve[i+1]*config.initial_capital),
                cash=cash, fees=fee_sum, actual_exposure=1-cash/after,
                actual_turnover=turnover, actual_max_single_weight=max(position['weights'], default=0.))))
    if not np.isclose(cash, result.final_cash, rtol=1e-12, atol=1e-7):
        raise ValueError('native final cash does not reconcile')
    return replace(result, trades=trades, construction_diagnostics=diagnostics)
