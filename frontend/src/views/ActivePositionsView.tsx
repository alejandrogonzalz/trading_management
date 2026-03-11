import { useMemo, useState, useEffect } from 'react';
import { LayoutDashboard, ShieldAlert, XCircle, ExternalLink, Activity, TrendingUp, TrendingDown, Target, Shield, Zap, AlertTriangle, History as HistoryIcon, ArrowUpRight, ArrowDownRight, Wallet } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const SmartTradeCard = ({ trade, onCancel, currentPrice }) => {
    const [isClosing, setIsClosing] = useState(false);
    
    const entryPrice = trade.smart_meta?.entry_price || parseFloat(trade.price) || 0;
    const tpPrice = trade.smart_meta?.tp || 0;
    const slPrice = trade.smart_meta?.sl || 0;
    const tradeSide = trade.smart_meta?.side || trade.side;
    const pnl = currentPrice > 0 ? ((currentPrice - entryPrice) / entryPrice * 100 * (tradeSide === 'BUY' ? 1 : -1)) : 0;
    
    const getProgress = () => {
        if (!currentPrice || !entryPrice || !tpPrice || !slPrice) return 50;
        const totalRange = tpPrice - slPrice;
        const currentPos = currentPrice - slPrice;
        return Math.min(Math.max((currentPos / totalRange) * 100, 0), 100);
    };

    const progress = getProgress();

    return (
        <div className="bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden hover:border-blue-500/30 transition-all group">
            <div className="p-6">
                <div className="flex justify-between items-start mb-6">
                    <div className="flex items-center gap-3">
                        <div className="p-3 bg-blue-600/10 rounded-2xl">
                            <Activity size={20} className="text-blue-400" />
                        </div>
                        <div>
                            <h3 className="text-xl font-black text-white tracking-tighter uppercase">{trade.symbol}</h3>
                            <div className="flex items-center gap-2">
                                <span className={`text-[10px] font-black px-2 py-0.5 rounded ${tradeSide === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                    {tradeSide === 'BUY' ? 'LONG' : 'SHORT'}
                                </span>
                                <span className="text-[10px] text-slate-500 font-bold font-mono">{trade.clientOrderId}</span>
                            </div>
                        </div>
                    </div>
                    <div className="text-right">
                        <p className={`text-2xl font-black font-mono tracking-tighter ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                        </p>
                        <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Unrealized P&L</p>
                    </div>
                </div>

                <div className="space-y-6">
                    <div className="relative pt-6 pb-2">
                        <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800/50">
                            <div 
                                className={`h-full transition-all duration-1000 ${pnl >= 0 ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]'}`}
                                style={{ width: `${progress}%` }}
                            ></div>
                        </div>
                        <div className="absolute top-0 w-full flex justify-between">
                            <div className="flex flex-col items-center"><span className="text-[9px] font-black text-rose-500">SL</span><span className="text-[10px] font-mono text-slate-500">${slPrice.toLocaleString()}</span></div>
                            <div className="flex flex-col items-center absolute" style={{ left: '50%', transform: 'translateX(-50%)' }}><span className="text-[9px] font-black text-slate-500 uppercase">Entry</span><span className="text-[10px] font-mono text-slate-400">${entryPrice.toLocaleString()}</span></div>
                            <div className="flex flex-col items-center"><span className="text-[9px] font-black text-emerald-500">TP</span><span className="text-[10px] font-mono text-slate-500">${tpPrice.toLocaleString()}</span></div>
                        </div>
                        <div className="absolute top-4 transition-all duration-1000 flex flex-col items-center" style={{ left: `${progress}%`, transform: 'translateX(-50%)' }}>
                            <div className="w-3 h-3 bg-white rounded-full border-2 border-slate-950 shadow-lg mb-1 z-10"></div>
                            <span className="text-[10px] font-black text-white bg-slate-800 px-2 py-0.5 rounded border border-slate-700 shadow-xl">${currentPrice.toLocaleString()}</span>
                        </div>
                    </div>
                </div>
            </div>
            <div className="p-4 bg-slate-900/30 border-t border-slate-800 flex gap-2">
                <button className="flex-1 bg-slate-800 hover:bg-slate-700 text-white text-[10px] font-black py-2.5 rounded-xl uppercase tracking-widest flex items-center justify-center gap-2"><Zap size={14} className="text-blue-400" />Break-Even</button>
                <button 
                    onClick={async () => { setIsClosing(true); await onCancel(trade.orderId, trade.symbol, trade); setIsClosing(false); }}
                    disabled={isClosing}
                    className={`flex-1 ${isClosing ? 'bg-slate-800 text-slate-500' : 'bg-rose-600/10 hover:bg-rose-600 text-rose-500 hover:text-white'} text-[10px] font-black py-2.5 rounded-xl transition-all uppercase tracking-widest border border-rose-500/20 hover:border-rose-600 flex items-center justify-center gap-2`}
                >
                    {isClosing ? <div className="w-3 h-3 border-2 border-slate-600 border-t-rose-400 rounded-full animate-spin"></div> : <XCircle size={14} />}
                    Panic Sell
                </button>
            </div>
        </div>
    );
};

const SmartHistoryTable = ({ history }) => {
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
                        <th className="p-6 text-center">Gross P&L</th>
                        <th className="p-6 text-center">Net Profit</th>
                        <th className="p-6 text-center">Closed At</th>
                    </tr>
                </thead>
                <tbody className="font-mono text-sm text-slate-300">
                    {history.length > 0 ? history.map(h => {
                        const grossPnl = ((h.exit_price - h.entry_price) / h.entry_price * 100 * (h.side === 'BUY' ? 1 : -1));
                        
                        // Estimate fees in dollar value
                        const entryFeeValue = h.fee_asset === h.symbol.replace('USDT','').replace('USDC','') ? (h.entry_fees * h.entry_price) : h.entry_fees;
                        const exitFeeValue = h.exit_fees; // Usually USDT for sells
                        const totalFees = (entryFeeValue || 0) + (exitFeeValue || 0);
                        
                        const invested = h.quantity * h.entry_price;
                        const grossProfit = invested * (grossPnl / 100);
                        const netProfit = grossProfit - totalFees;
                        const netPnlPercent = (netProfit / invested) * 100;

                        return (
                            <tr key={h.id} className="border-b border-slate-800/30 hover:bg-slate-800/10 transition-all">
                                <td className="p-6 font-black text-white">{h.symbol}</td>
                                <td className="p-6 text-center">
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-black ${h.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                        {h.side === 'BUY' ? 'LONG' : 'SHORT'}
                                    </span>
                                </td>
                                <td className="p-6 text-center">${Number(h.entry_price || 0).toLocaleString()}</td>
                                <td className="p-6 text-center">${Number(h.exit_price || 0).toLocaleString()}</td>
                                <td className="p-6 text-center text-slate-500">-${totalFees.toFixed(4)}</td>
                                <td className="p-6 text-center">
                                    <div className={`font-black ${grossPnl >= 0 ? 'text-emerald-500/60' : 'text-rose-500/60'}`}>
                                        {grossPnl.toFixed(2)}%
                                    </div>
                                </td>
                                <td className="p-6 text-center">
                                    <div className={`flex items-center justify-center gap-1 font-black ${netProfit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        {netProfit >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                                        {netPnlPercent.toFixed(2)}%
                                        <span className="text-[10px] ml-1 opacity-60">(${netProfit.toFixed(2)})</span>
                                    </div>
                                </td>
                                <td className="p-6 text-center text-slate-500 text-[10px]">
                                    {new Date(h.close_time * 1000).toLocaleString()}
                                </td>
                            </tr>
                        );
                    }) : (
                        <tr><td colSpan={8} className="py-32 text-center text-slate-700 uppercase font-black opacity-30 tracking-widest">No Closed Trades Yet</td></tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}

const ActivePositionsView = ({ openOrders, handleCancelOrder, quoteBalance }) => {
    const [prices, setPrices] = useState({});
    const [activeTab, setActiveTab] = useState('active'); // 'active' or 'history'
    const [smartHistory, setSmartHistory] = useState([]);

    useEffect(() => {
        if (activeTab === 'history') {
            fetch(`${API_BASE}/trades/smart-history`)
                .then(res => res.json())
                .then(data => setSmartHistory(data))
                .catch(err => console.error("History fetch failed:", err));
        }
    }, [activeTab, openOrders]);

    useEffect(() => {
        const smartOrders = openOrders.filter(o => (o.clientOrderId?.startsWith('SMART_') || o.listClientOrderId?.startsWith('LIST_SMART_')));
        if (smartOrders.length === 0) return;
        const fetchPrices = async () => {
            const newPrices = { ...prices };
            for (const order of smartOrders) {
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
    }, [openOrders]);
    
    const smartTrades = useMemo(() => {
        const tradesMap = new Map();
        openOrders.forEach(o => {
            const smartMeta = (o as any).smart_meta;
            if (smartMeta) {
                let masterId = null;
                if (o.orderId.toString().startsWith('POS_')) masterId = o.orderId.toString().replace('POS_', '');
                else if (o.clientOrderId?.startsWith('SMART_')) masterId = o.clientOrderId;
                else if (o.listClientOrderId?.startsWith('LIST_SMART_')) masterId = o.listClientOrderId.replace('LIST_', '');
                if (masterId && (!tradesMap.has(masterId) || o.type === 'POSITION')) tradesMap.set(masterId, o);
            }
        });
        return Array.from(tradesMap.values());
    }, [openOrders]);

    const onCancelClick = async (orderId: any, symbol: string, trade?: any) => {
        if (orderId.toString().startsWith('POS_') || trade?.smart_meta) {
            if (window.confirm(`Do you want to Market Sell the ${trade?.origQty || trade?.smart_meta?.quantity || ''} ${symbol} associated with this Smart Trade?`)) {
                try {
                    const res = await fetch(`${API_BASE}/trades/market-close`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ symbol, quantity: trade?.origQty || trade?.smart_meta?.quantity, orderListId: trade?.smart_meta?.orderListId })
                    });
                    if (res.ok) console.log('Smart Position Closed!');
                } catch (e) { console.error('Error during market close:', e); }
            }
            return;
        }
        if (window.confirm(`Are you sure you want to cancel order #${orderId} for ${symbol}?`)) handleCancelOrder(orderId, symbol);
    };

    return (
        <div className="h-full flex flex-col bg-slate-900 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-800">
            <header className="px-6 pt-8 pb-4">
                <div className="flex justify-between items-end">
                    <div>
                        <h1 className="text-3xl font-black text-white flex items-center gap-3 tracking-tighter uppercase">
                            <Zap size={32} className="text-blue-500" /> SMART TRADES
                        </h1>
                        <div className="flex gap-1 mt-4">
                            <button onClick={() => setActiveTab('active')} className={`px-6 py-2 rounded-t-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'active' ? 'bg-slate-800 text-blue-400 border-b-2 border-blue-500' : 'text-slate-500 hover:text-slate-300'}`}>Active Setups</button>
                            <button onClick={() => setActiveTab('history')} className={`px-6 py-2 rounded-t-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'history' ? 'bg-slate-800 text-blue-400 border-b-2 border-blue-500' : 'text-slate-500 hover:text-slate-300'}`}>Closed Trades</button>
                        </div>
                    </div>
                    <div className="bg-slate-950 px-6 py-3 rounded-2xl border border-slate-800 flex flex-col items-end shadow-lg mb-2">
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1 opacity-60">Total Cash</span>
                        <p className="font-mono text-xl font-black text-white tracking-tighter">${parseFloat(quoteBalance).toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                    </div>
                </div>
                <div className="h-px w-full bg-slate-800"></div>
            </header>

            <div className="px-6 py-6">
                {activeTab === 'active' ? (
                    smartTrades.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                            {smartTrades.map(trade => (
                                <SmartTradeCard key={trade.orderId} trade={trade} onCancel={onCancelClick} currentPrice={prices[trade.symbol] || 0} />
                            ))}
                        </div>
                    ) : (
                        <div className="py-32 text-center bg-slate-950 rounded-3xl border border-slate-800 border-dashed opacity-20">
                            <Zap size={64} className="mx-auto mb-4" />
                            <p className="font-black text-xl tracking-widest uppercase">No Active Smart Trades</p>
                        </div>
                    )
                ) : (
                    <SmartHistoryTable history={smartHistory} />
                )}
            </div>
        </div>
    );
};

export default ActivePositionsView;
