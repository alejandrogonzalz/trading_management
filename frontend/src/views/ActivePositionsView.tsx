import { useMemo, useState, useEffect } from 'react';
import { Activity, XCircle, TrendingUp, Zap, ArrowUpRight, ArrowDownRight, ShieldCheck, Target, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const SmartTradeCard = ({ trade, onCancel, currentPrice, onSelectSymbol, leadPositions }) => {
    const [isClosing, setIsClosing] = useState(false);
    const navigate = useNavigate();

    const handleSymbolClick = () => {
        onSelectSymbol(trade.symbol);
        navigate('/');
    };
    
    // Futures identification
    const leverage = trade.smart_meta?.leverage || trade.leverage;
    const isFutures = !!leverage;
    
    const livePos = leadPositions?.find(p => p.symbol === trade.symbol);
    
    // Metrics
    const entryPrice = livePos ? parseFloat(livePos.entryPrice) : 
                       (parseFloat(trade.entry_price) || parseFloat(trade.smart_meta?.entry_price) || parseFloat(trade.price) || 0);
                       
    const quantity = livePos ? Math.abs(parseFloat(livePos.positionAmt)) : 
                     (parseFloat(trade.origQty) || trade.smart_meta?.quantity || 0);

    const liqPrice = livePos ? parseFloat(livePos.liquidationPrice) : 0;
    const markPrice = livePos ? parseFloat(livePos.markPrice) : currentPrice;
    const unrealizedProfit = livePos ? parseFloat(livePos.unRealizedProfit) : null;

    const tradeSide = trade.smart_meta?.side || trade.side;
    const tpPrice = Number(trade.smart_meta?.tp || trade.tp) || 0;
    const slPrice = Number(trade.smart_meta?.sl || trade.sl) || 0;
    const needsProtection = trade.needs_protection === true;
    
    // For progress bar when TP/SL are missing (raw position)
    const isRawPosition = !tpPrice && !slPrice;
    
    const isSyncing = !currentPrice || currentPrice === 0;

    // P&L Logic
    let netProfit, netPnlPercent;
    
    if (isFutures && unrealizedProfit !== null) {
        // Use live futures data if available
        netProfit = unrealizedProfit;
        const margin = (quantity * entryPrice) / (leverage || 1);
        netPnlPercent = margin > 0 ? (netProfit / margin) * 100 : 0;
    } else {
        // Fallback to manual calculation
        const grossProfit = entryPrice > 0 ? (currentPrice - entryPrice) * quantity * (tradeSide === 'BUY' || tradeSide === 'LONG' ? 1 : -1) : 0;
        const entryFees = trade.smart_meta?.entry_fees || 0;
        const estExitFee = Math.abs(grossProfit + (entryPrice * quantity)) * 0.001; 
        netProfit = grossProfit - entryFees - estExitFee;
        
        if (isFutures) {
            const margin = (quantity * entryPrice) / (leverage || 1);
            netPnlPercent = margin > 0 ? (netProfit / margin) * 100 : 0;
        } else {
            netPnlPercent = (quantity * entryPrice) > 0 ? (netProfit / (quantity * entryPrice) * 100) : 0;
        }
    }
    
    const getPos = (price, tp, sl) => {
        if (!price || !tp || !sl || tp === sl) return 50;
        const totalRange = Math.abs(tp - sl);
        const currentDiff = Math.abs(price - sl);
        const pos = (currentDiff / totalRange) * 100;
        return Math.min(Math.max(pos, 0), 100);
    };

    // Enhanced position calculation for raw futures positions
    const getRawPosition = (currentPrice, entryPrice, liqPrice, side) => {
        if (!liqPrice || !entryPrice || entryPrice === liqPrice) return 50;
        
        const isLong = side === 'BUY';
        let minPrice, maxPrice;
        
        if (isLong) {
            // LONG: liquidation is below entry, profit is above entry
            // We show from liquidation (0%) to entry + (entry - liq) (100%)
            // Entry is at 50%
            minPrice = liqPrice;
            maxPrice = entryPrice + (entryPrice - liqPrice); // Symmetrical range
            if (currentPrice < minPrice) return 0;
            if (currentPrice > maxPrice) return 100;
            const range = maxPrice - minPrice;
            return ((currentPrice - minPrice) / range) * 100;
        } else {
            // SHORT: liquidation is above entry, profit is below entry  
            // We show from entry - (liq - entry) (0%) to liquidation (100%)
            // Entry is at 50%
            maxPrice = liqPrice;
            minPrice = entryPrice - (liqPrice - entryPrice); // Symmetrical range
            if (currentPrice < minPrice) return 100; // Beyond profit range (far left)
            if (currentPrice > maxPrice) return 0;   // Beyond liquidation (far right)
            const range = maxPrice - minPrice;
            return ((maxPrice - currentPrice) / range) * 100; // Invert: profit is left
        }
    };

    const getRawEntryPosition = (entryPrice, liqPrice, side) => {
        if (!liqPrice || !entryPrice || entryPrice === liqPrice) return 50;
        const isLong = side === 'BUY';
        
        if (isLong) {
            const minPrice = liqPrice;
            const maxPrice = entryPrice + (entryPrice - liqPrice);
            const range = maxPrice - minPrice;
            return ((entryPrice - minPrice) / range) * 100;
        } else {
            const maxPrice = liqPrice;
            const minPrice = entryPrice - (liqPrice - entryPrice);
            const range = maxPrice - minPrice;
            return ((maxPrice - entryPrice) / range) * 100;
        }
    };

    // For Raw Positions, use enhanced visualization
    const entryMarkerPos = isRawPosition ? 
        (liqPrice > 0 ? getRawEntryPosition(entryPrice, liqPrice, tradeSide) : 50) : 
        getPos(entryPrice, tpPrice, slPrice);
        
    const currentMarkerPos = isSyncing ? entryMarkerPos : (
        isRawPosition ? 
            (liqPrice > 0 ? getRawPosition(markPrice, entryPrice, liqPrice, tradeSide) : 50) : 
            getPos(currentPrice, tpPrice, slPrice)
    );

    const fmt = (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: val < 1 ? 6 : 2, maximumFractionDigits: val < 1 ? 8 : 2 });

    const theme = isFutures ? {
        border: 'hover:border-orange-500/30',
        icon: 'text-orange-400',
        iconBg: 'bg-orange-600/10'
    } : {
        border: 'hover:border-blue-500/30',
        icon: 'text-blue-400',
        iconBg: 'bg-blue-600/10'
    };

    return (
        <div className={`bg-slate-950 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden ${theme.border} transition-all group w-full mb-4`}>
            <div className="flex flex-col xl:flex-row">
                {/* Symbol Section */}
                <div className="p-5 flex flex-col justify-center border-b xl:border-b-0 xl:border-r border-slate-800 min-w-[220px] bg-slate-900/20">
                    <div className="flex items-center gap-3 mb-2">
                        <div className={`p-2 ${theme.iconBg} rounded-lg`}>
                            <Activity size={18} className={theme.icon} />
                        </div>
                        <button 
                            onClick={handleSymbolClick}
                            className={`text-lg font-black text-white tracking-tighter uppercase hover:${theme.icon} transition-colors hover:underline`}
                        >
                            {trade.symbol}
                        </button>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className={`text-[9px] font-black px-2 py-0.5 rounded ${tradeSide === "BUY" || tradeSide === "LONG" ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
                            {tradeSide === "BUY" || tradeSide === "LONG" ? "LONG" : "SHORT"} {isFutures ? `${leverage}x` : ""}
                        </span>
                        <span className="text-[9px] text-slate-600 font-bold font-mono">{isRawPosition ? "RAW POSITION" : (isFutures ? "SMART LEAD" : "SMART SPOT")}</span>
                        {needsProtection && (
                            <span className="text-[9px] font-black px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 flex items-center gap-1">
                                <AlertCircle size={9} />
                                Needs Protection
                            </span>
                        )}
                    </div>
                </div>

                {/* Progress Visualizer */}
                <div className="flex-1 p-6 flex flex-col justify-center space-y-8">
                    <div className="relative pt-2">
                        <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800/50 relative">
                            <div 
                                className={`absolute h-full transition-all duration-1000 ${netProfit >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                style={{ 
                                    left: `${Math.min(entryMarkerPos, currentMarkerPos)}%`, 
                                    width: `${Math.abs(currentMarkerPos - entryMarkerPos)}%` 
                                }}
                            ></div>
                        </div>
                        <div className="absolute top-[-12px] w-full text-[8px] font-black uppercase tracking-tighter text-slate-600">
                            {isRawPosition ? (
                                <>
                                    <span className="absolute left-0 text-rose-500">Liquidation</span>
                                    <span className="absolute right-0 text-emerald-500">Entry</span>
                                </>
                            ) : (
                                <>
                                    <span className="absolute left-0 text-rose-500">Stop Loss</span>
                                    <span className="absolute" style={{ left: `${entryMarkerPos}%`, transform: 'translateX(-50%)' }}>Entry</span>
                                    <span className="absolute right-0 text-emerald-500">Take Profit</span>
                                </>
                            )}
                        </div>
                        <div 
                            className={`absolute top-[-2px] transition-all duration-1000 flex flex-col items-center z-10 ${isSyncing ? 'animate-pulse opacity-50' : ''}`}
                            style={{ left: `${currentMarkerPos}%`, transform: 'translateX(-50%)' }}
                        >
                            <div className="w-2.5 h-2.5 bg-white rounded-full border-2 border-slate-950 shadow-xl mb-1"></div>
                            <span className="text-[10px] font-black text-white bg-slate-800 px-2 py-0.5 rounded border border-slate-700 shadow-2xl whitespace-nowrap">
                                {isSyncing ? 'Syncing...' : `$${fmt(currentPrice)}`}
                            </span>
                        </div>
                    </div>
                    
                    {/* Metrics Grid - Consistent 3-column layout like Spot trades */}
                    <div className="grid grid-cols-3 gap-4 text-center">
                        <div className="flex flex-col">
                            <span className="text-[10px] font-black text-rose-500/50 uppercase tracking-widest mb-1">SL Level</span>
                            <span className="text-sm font-mono font-bold text-slate-400">
                                {slPrice > 0 ? `$${fmt(slPrice)}` : (isFutures && liqPrice > 0 ? `$${fmt(liqPrice)}` : 'N/A')}
                                {isFutures && liqPrice > 0 && slPrice <= 0 && (
                                    <span className="text-[8px] text-slate-500 block">Liquidation</span>
                                )}
                            </span>
                        </div>
                        <div className="flex flex-col border-x border-slate-800/50">
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Avg Entry</span>
                            <span className="text-sm font-mono font-bold text-slate-200">${fmt(entryPrice)}</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-[10px] font-black text-emerald-500/50 uppercase tracking-widest mb-1">TP Target</span>
                            <span className="text-sm font-mono font-bold text-slate-400">
                                {tpPrice > 0 ? `$${fmt(tpPrice)}` : 'N/A'}
                                {isFutures && tpPrice <= 0 && !isRawPosition && (
                                    <span className="text-[8px] text-slate-500 block">No TP set</span>
                                )}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Action Section */}
                <div className="p-5 flex flex-col xl:flex-row items-center gap-6 border-t xl:border-t-0 xl:border-l border-slate-800 bg-slate-900/10 min-w-[320px]">
                    <div className="text-right flex-1 w-full xl:w-auto">
                        {isSyncing ? (
                            <div className="flex flex-col items-end gap-1 animate-pulse">
                                <div className="h-8 w-24 bg-slate-800 rounded-lg mb-1"></div>
                                <div className="h-4 w-32 bg-slate-800/50 rounded-lg"></div>
                            </div>
                        ) : (
                            <>
                                <p className={`text-3xl font-black font-mono tracking-tighter ${netProfit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {netProfit >= 0 ? '+' : ''}{netPnlPercent.toFixed(2)}%
                                </p>
                                <div className="flex items-center justify-end gap-2 mt-1">
                                    <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Net Profit:</span>
                                    <span className={`text-lg font-mono font-black ${netProfit >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                        {netProfit >= 0 ? '+' : ''}${netProfit.toFixed(2)}
                                    </span>
                                </div>
                            </>
                        )}
                    </div>
                    <div className="flex gap-2 w-full xl:w-auto">
                        <button 
                            onClick={async () => { setIsClosing(true); await onCancel(trade.orderId || trade.id, trade.symbol, trade); setIsClosing(false); }}
                            disabled={isClosing}
                            className={`flex-1 xl:flex-none px-6 py-2.5 rounded-xl font-black uppercase tracking-widest text-[10px] transition-all flex items-center justify-center gap-2 ${isClosing ? 'bg-slate-800 text-slate-500' : 'bg-rose-600/10 hover:bg-rose-600 text-rose-500 hover:text-white border border-rose-500/20 hover:border-rose-600'}`}
                        >
                            {isClosing ? <div className="w-3 h-3 border-2 border-slate-600 border-t-rose-400 rounded-full animate-spin"></div> : <XCircle size={14} />}
                            Panic Sell
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

const SmartHistoryTable = ({ history, onSelectSymbol, mode }) => {
    const isLead = mode === 'LEAD';
    const fmt = (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: val < 1 ? 6 : 2, maximumFractionDigits: val < 1 ? 8 : 2 });
    const navigate = useNavigate();
    
    const getDuration = (start, end) => {
        if (!start || !end) return 'N/A';
        const diff = end - start;
        const hours = Math.floor(diff / 3600);
        const mins = Math.floor((diff % 3600) / 60);
        if (hours > 0) return `${hours}h ${mins}m`;
        return `${mins}m`;
    };

    return (
        <div className="bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden">
            <table className="w-full text-left border-collapse">
                <thead className="bg-slate-900">
                    <tr className="text-slate-500 text-[10px] uppercase font-black tracking-widest border-b border-slate-800">
                        <th className="p-6">Symbol</th>
                        <th className="p-6 text-center">Side</th>
                        <th className="p-6 text-center">Entry Price</th>
                        <th className="p-6 text-center">Exit Price</th>
                        <th className="p-6 text-center">Fees</th>
                        <th className="p-6 text-center">Net P&L</th>
                        <th className="p-6 text-center">Opened At</th>
                        <th className="p-6 text-center">Closed At</th>
                        <th className="p-6 text-center">Duration</th>
                    </tr>
                </thead>
                <tbody className="font-mono text-xs text-slate-300">
                    {history.length > 0 ? history.map(h => {
                        if (isLead) {
                            const pnl = ((h.exit_price - h.entry_price) / h.entry_price * 100 * (h.side === 'BUY' || h.side === 'LONG' ? 1 : -1));
                            const grossPnl = pnl * (h.leverage || 1);
                            
                            const totalFees = h.exit_fees || 0; 
                            const feeAsset = h.exit_fee_asset || 'USDT';
                            
                            const initialMargin = (h.quantity * h.entry_price) / (h.leverage || 1);
                            const grossProfitVal = (initialMargin * (grossPnl / 100));
                            const netProfitVal = grossProfitVal - totalFees;
                            const netPnlPercent = initialMargin > 0 ? (netProfitVal / initialMargin * 100) : 0;

                            return (
                                <tr key={h.id || h._id} className="border-b border-slate-800/30 hover:bg-slate-800/10 transition-all">
                                    <td className="p-6">
                                        <button onClick={() => { onSelectSymbol(h.symbol); navigate('/'); }} className="font-black text-white hover:text-orange-400 transition-colors hover:underline text-left">
                                            {h.symbol}
                                        </button>
                                    </td>
                                    <td className="p-6 text-center">
                                        <div className="flex flex-col items-center gap-1">
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-black ${h.side === 'BUY' || h.side === 'LONG' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                                {h.side === 'BUY' || h.side === 'LONG' ? 'LONG' : 'SHORT'}
                                            </span>
                                            <span className="text-[9px] text-orange-400 font-bold">{h.leverage}x</span>
                                        </div>
                                    </td>
                                    <td className="p-6 text-center">${fmt(h.entry_price)}</td>
                                    <td className="p-6 text-center">${fmt(h.exit_price)}</td>
                                    <td className="p-6 text-center">
                                        <span className="text-slate-500 text-[10px] font-bold">
                                            -{totalFees.toFixed(4)} <span className="text-[8px] opacity-70">{feeAsset}</span>
                                        </span>
                                    </td>
                                    <td className="p-6 text-center">
                                        <div className={`flex flex-col items-center font-black ${netProfitVal >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            <div className="flex items-center gap-1">
                                                {netProfitVal >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                                                {netPnlPercent.toFixed(2)}%
                                            </div>
                                            <span className="text-[10px] opacity-60">(${netProfitVal.toFixed(2)})</span>
                                        </div>
                                    </td>
                                    <td className="p-6 text-center text-slate-500 text-[10px]">
                                        {new Date((h.timestamp || h.time/1000) * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                                    </td>
                                    <td className="p-6 text-center text-slate-500 text-[10px]">
                                        {new Date((h.close_time || h.time) * (h.close_time < 10000000000 ? 1000 : 1)).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                                    </td>
                                    <td className="p-6 text-center">
                                        <span className="bg-slate-900 px-2 py-1 rounded text-[10px] font-bold text-slate-400">
                                            {getDuration(h.timestamp || h.time/1000, (h.close_time < 10000000000 ? h.close_time : h.close_time/1000))}
                                        </span>
                                    </td>
                                </tr>
                            );
                        }

                        const assetName = h.symbol.replace('USDT','').replace('USDC','');
                        let entryFeeValue = h.fee_asset === assetName ? (h.entry_fees * h.entry_price) : (h.fee_asset === 'BNB' ? h.entry_fees * 600 : (h.entry_fees || 0));
                        let exitFeeValue = h.exit_fee_asset === assetName ? (h.exit_fees * h.exit_price) : (h.exit_fee_asset === 'BNB' ? h.exit_fees * 600 : (h.exit_fees || 0));
                        const totalFees = entryFeeValue + exitFeeValue;
                        const grossPnl = ((h.exit_price - h.entry_price) / h.entry_price * 100 * (h.side === 'BUY' ? 1 : -1));
                        const invested = h.quantity * h.entry_price;
                        const grossProfit = invested * (grossPnl / 100);
                        const netProfit = grossProfit - totalFees;
                        const netPnlPercent = (netProfit / invested) * 100;

                        return (
                            <tr key={h.id || h._id} className="border-b border-slate-800/30 hover:bg-slate-800/10 transition-all">
                                <td className="p-6">
                                    <button onClick={() => { onSelectSymbol(h.symbol); navigate('/'); }} className="font-black text-white hover:text-blue-400 transition-colors hover:underline text-left">
                                        {h.symbol}
                                    </button>
                                </td>
                                <td className="p-6 text-center">
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-black ${h.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                        {h.side === 'BUY' ? 'LONG' : 'SHORT'}
                                    </span>
                                </td>
                                <td className="p-6 text-center">${fmt(h.entry_price)}</td>
                                <td className="p-6 text-center">${fmt(h.exit_price)}</td>
                                <td className="p-6 text-center text-slate-500">-${totalFees.toFixed(4)}</td>
                                <td className="p-6 text-center">
                                    <div className={`flex flex-col items-center font-black ${netProfit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        <div className="flex items-center gap-1">
                                            {netProfit >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                                            {netPnlPercent.toFixed(2)}%
                                        </div>
                                        <span className="text-[10px] opacity-60">(${netProfit.toFixed(2)})</span>
                                    </div>
                                </td>
                                <td className="p-6 text-center text-slate-500 text-[10px]">
                                    {new Date(h.timestamp * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                                </td>
                                <td className="p-6 text-center text-slate-500 text-[10px]">
                                    {new Date(h.close_time * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                                </td>
                                <td className="p-6 text-center">
                                    <span className="bg-slate-900 px-2 py-1 rounded text-[10px] font-bold text-slate-400">
                                        {getDuration(h.timestamp, h.close_time)}
                                    </span>
                                </td>
                            </tr>
                        );
                    }) : (
                        <tr><td colSpan={9} className="py-32 text-center text-slate-700 uppercase font-black opacity-30 tracking-widest">No Closed Trades Yet</td></tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}

const ActivePositionsView = ({ openOrders, handleCancelOrder, onSelectSymbol, tradingMode, leadPositions, smartHistory }) => {
    const isLead = tradingMode === 'LEAD';
    const [prices, setPrices] = useState({});
    const [activeTab, setActiveTab] = useState('active');

    const stats = useMemo(() => {
        if (!smartHistory || smartHistory.length === 0) return null;
        
        let totalNetProfit = 0;
        let wins = 0;
        let bestTrade = -Infinity;
        
        const historyWithNet = smartHistory.map(h => {
            if (isLead) {
                const pnl = ((h.exit_price - h.entry_price) / h.entry_price * (h.side === 'BUY' || h.side === 'LONG' ? 1 : -1));
                const netProfit = pnl * (h.quantity * h.entry_price) * (h.leverage || 1) - (h.exit_fees || 0);
                const netPnlPercent = pnl * 100 * (h.leverage || 1);
                return { netProfit, netPnlPercent };
            }

            const assetName = h.symbol.replace('USDT','').replace('USDC','');
            let entryFeeValue = h.fee_asset === assetName ? (h.entry_fees * h.entry_price) : (h.fee_asset === 'BNB' ? h.entry_fees * 600 : (h.entry_fees || 0));
            let exitFeeValue = h.exit_fee_asset === assetName ? (h.exit_fees * h.exit_price) : (h.exit_fee_asset === 'BNB' ? h.exit_fees * 600 : (h.exit_fees || 0));
            
            const tradeValue = h.quantity * h.entry_price;
            const totalFees = entryFeeValue + exitFeeValue;

            const grossProfit = (h.exit_price - h.entry_price) * h.quantity * (h.side === 'BUY' ? 1 : -1);
            const netProfit = grossProfit - totalFees;
            const netPnlPercent = tradeValue > 0 ? (netProfit / tradeValue) * 100 : 0;
            
            return { netProfit, netPnlPercent };
        });

        historyWithNet.forEach(h => {
            totalNetProfit += h.netProfit;
            if (h.netProfit > 0) wins++;
            if (h.netPnlPercent > bestTrade) bestTrade = h.netPnlPercent;
        });

        return {
            totalProfit: totalNetProfit,
            winRate: (wins / smartHistory.length) * 100,
            avgProfit: totalNetProfit / smartHistory.length,
            bestTrade: bestTrade === -Infinity ? 0 : bestTrade,
            count: smartHistory.length
        };
    }, [smartHistory, isLead]);

    const leadSmartTrades = useMemo(() => {
        if (!isLead) return [];
        // Lead smart trades have leverage in smart_meta or ID starts with LEAD_
        return openOrders.filter(o => 
            (o as any).smart_meta?.leverage || 
            o.orderId?.toString().startsWith('LEAD_') || 
            o.clientOrderId?.startsWith('LEAD_')
        );
    }, [openOrders, isLead]);

    const leadRawPositions = useMemo(() => {
        if (!isLead) return [];
        // Only show raw positions that aren't already covered by a Smart Trade
        // We map them to look like trades so they can use SmartTradeCard
        return leadPositions
            .filter(p => !leadSmartTrades.some(t => t.symbol === p.symbol))
            .map(p => ({
                id: `POS_${p.symbol}`,
                orderId: `POS_${p.symbol}`,
                symbol: p.symbol,
                side: parseFloat(p.positionAmt) > 0 ? 'LONG' : 'SHORT',
                leverage: parseInt(p.leverage),
                entry_price: parseFloat(p.entryPrice),
                origQty: Math.abs(parseFloat(p.positionAmt)),
                type: 'POSITION'
            }));
    }, [leadPositions, leadSmartTrades, isLead]);

    const spotSmartTrades = useMemo(() => {
        if (isLead) return [];
        const tradesMap = new Map();
        openOrders.forEach(o => {
            const smartMeta = (o as any).smart_meta;
            const isSpotSmart = smartMeta && !smartMeta.leverage;

            if (isSpotSmart) {
                let masterId = null;
                if (o.orderId.toString().startsWith('POS_')) masterId = o.orderId.toString().replace('POS_', '');
                else if (o.clientOrderId?.startsWith('SMART_')) masterId = o.clientOrderId;
                else if (o.listClientOrderId?.startsWith('LIST_SMART_')) masterId = o.listClientOrderId.replace('LIST_', '');
                if (masterId && (!tradesMap.has(masterId) || o.type === 'POSITION')) tradesMap.set(masterId, o);
            }
        });
        return Array.from(tradesMap.values());
    }, [openOrders, isLead]);

    useEffect(() => {
        const setups = isLead ? [...leadSmartTrades, ...leadRawPositions] : spotSmartTrades;
        if (setups.length === 0) return;

        const fetchPrices = async () => {
            const newPrices = { ...prices };
            for (const order of setups) {
                try {
                    const res = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${order.symbol}`);
                    if (res.ok) { const data = await res.json(); newPrices[order.symbol] = parseFloat(data.price); }
                } catch (e) {}
            }
            setPrices(newPrices);
        };
        fetchPrices();
        const interval = setInterval(fetchPrices, 3000);
        return () => clearInterval(interval);
    }, [leadSmartTrades, leadRawPositions, spotSmartTrades, isLead]);

    const onCancelClick = async (orderId: any, symbol: string, trade?: any) => {
        const isLeadTrade = isLead || orderId.toString().startsWith('LEAD_') || (trade as any)?.smart_meta?.leverage || (trade as any)?.leverage;

        if (orderId.toString().startsWith('POS_') || trade?.smart_meta || trade?.type === 'POSITION') {
            const qty = trade?.origQty || trade?.smart_meta?.quantity || 0;
            if (window.confirm(`Market Sell the ${qty} ${symbol}?`)) {
                try {
                    const endpoint = isLeadTrade ? `${API_BASE}/lead/close-position` : `${API_BASE}/trades/market-close`;
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ symbol, quantity: qty })
                    });
                    if (res.ok) console.log('Closed!');
                } catch (e) { console.error(e); }
            }
            return;
        }
        if (window.confirm(`Cancel order for ${symbol}?`)) handleCancelOrder(orderId, symbol);
    };

    const theme = isLead ? {
        text: 'text-orange-500',
        activeText: 'text-orange-400',
        activeBorder: 'border-orange-500',
        bg: 'bg-orange-600/5',
        icon: 'text-orange-500'
    } : {
        text: 'text-blue-500',
        activeText: 'text-blue-400',
        activeBorder: 'border-blue-500',
        bg: 'bg-blue-600/5',
        icon: 'text-blue-500'
    };

    return (
        <div className="h-full flex flex-col bg-slate-900 overflow-hidden text-slate-100">
            <header className="px-6 pt-8 pb-4 shrink-0">
                <div className="max-w-[1400px] w-full flex justify-between items-start">
                    <div>
                        <h1 className="text-3xl font-black text-white flex items-center gap-3 tracking-tighter uppercase">
                            <Zap size={32} className={theme.icon} /> 
                            {isLead ? 'SMART LEAD TRADES' : 'SMART SPOT TRADES'}
                        </h1>
                        <div className="flex gap-1 mt-4">
                            <button onClick={() => setActiveTab('active')} className={`px-6 py-2 rounded-t-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'active' ? `bg-slate-800 ${theme.activeText} border-b-2 ${theme.activeBorder}` : 'text-slate-500 hover:text-slate-300'}`}>
                                {isLead ? 'Active Positions' : 'Active Setups'}
                            </button>
                            <button onClick={() => setActiveTab('history')} className={`px-6 py-2 rounded-t-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'history' ? `bg-slate-800 ${theme.activeText} border-b-2 ${theme.activeBorder}` : 'text-slate-500 hover:text-slate-300'}`}>
                                {isLead ? 'Smart Lead History' : 'Smart Spot History'}
                            </button>
                        </div>
                    </div>

                    {stats && (
                        <div className="flex gap-4 animate-in fade-in slide-in-from-right-4 duration-700">
                            <div className="bg-slate-950/50 border border-slate-800 rounded-2xl px-6 py-3 flex flex-col items-end min-w-[120px]">
                                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Win Rate</span>
                                <div className="flex items-center gap-2">
                                    <span className="text-lg font-black text-emerald-400 font-mono">{stats.winRate.toFixed(1)}%</span>
                                    <TrendingUp size={14} className="text-emerald-500/50" />
                                </div>
                            </div>
                            <div className="bg-slate-950/50 border border-slate-800 rounded-2xl px-6 py-3 flex flex-col items-end min-w-[120px]">
                                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Total Net P&L</span>
                                <span className={`text-lg font-black font-mono ${stats.totalProfit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {stats.totalProfit >= 0 ? '+' : ''}${stats.totalProfit.toFixed(2)}
                                </span>
                            </div>
                            <div className="bg-slate-950/50 border border-slate-800 rounded-2xl px-6 py-3 flex flex-col items-end min-w-[120px]">
                                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Avg per Trade</span>
                                <span className={`text-lg font-black font-mono ${stats.avgProfit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    ${stats.avgProfit.toFixed(2)}
                                </span>
                            </div>
                            <div className={`${theme.bg} border ${theme.activeBorder}/20 rounded-2xl px-6 py-3 flex flex-col items-end min-w-[120px]`}>
                                <span className={`text-[8px] font-black ${theme.activeText}/70 uppercase tracking-widest mb-1`}>Best Trade</span>
                                <span className={`text-lg font-black text-slate-100 font-mono`}>+{stats.bestTrade.toFixed(2)}%</span>
                            </div>
                        </div>
                    )}
                </div>
                <div className={`h-px max-w-[1400px] w-full bg-slate-800`}></div>
            </header>

            <div className="flex-1 overflow-y-auto px-6 py-2 custom-scrollbar">
                <div className="max-w-[1400px] w-full pb-12">
                    {activeTab === 'active' ? (
                        <div className="flex flex-col gap-4 w-full">
                            {isLead && [...leadSmartTrades, ...leadRawPositions].map(trade => (
                                <SmartTradeCard key={trade.orderId || trade.id} trade={trade} onCancel={onCancelClick} currentPrice={prices[trade.symbol] || 0} onSelectSymbol={onSelectSymbol} leadPositions={leadPositions} />
                            ))}
                            {!isLead && spotSmartTrades.map(trade => (
                                <SmartTradeCard key={trade.orderId || trade.id} trade={trade} onCancel={onCancelClick} currentPrice={prices[trade.symbol] || 0} onSelectSymbol={onSelectSymbol} />
                            ))}

                            {isLead && leadSmartTrades.length === 0 && leadRawPositions.length === 0 && (
                                <div className="py-32 text-center bg-slate-950 rounded-3xl border border-slate-800 border-dashed opacity-20">
                                    <ShieldCheck size={64} className="mx-auto mb-4" />
                                    <p className="font-black text-xl tracking-widest uppercase">No Active Lead Trades</p>
                                </div>
                            )}
                            {!isLead && spotSmartTrades.length === 0 && (
                                <div className="py-32 text-center bg-slate-950 rounded-3xl border border-slate-800 border-dashed opacity-20">
                                    <Zap size={64} className="mx-auto mb-4" />
                                    <p className="font-black text-xl tracking-widest uppercase">No Active Smart Setups</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <SmartHistoryTable history={smartHistory} onSelectSymbol={onSelectSymbol} mode={tradingMode} />
                    )}
                </div>
            </div>
        </div>
    );
};

export default ActivePositionsView;
