import { useState, useMemo, useEffect } from 'react';
import { ListFilter, LayoutDashboard, History } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const BottomPanel = ({ 
    openOrders = [], 
    tradeHistory = [], 
    symbol = '', 
    filterOrdersBySymbol = true, 
    setFilterOrdersBySymbol = () => {}, 
    handleCancelOrder = () => {},
    tradingMode = 'SPOT',
    leadPositions = []
}) => {
    const [activeTab, setActiveTab] = useState('orders');
    const [binanceHistory, setBinanceHistory] = useState([]);
    const [isLoadingHistory, setIsLoadingHistory] = useState(false);
    const isLead = tradingMode === 'LEAD';

    // Fetch raw exchange history when tab is clicked
    useEffect(() => {
        if (activeTab === 'history') {
            const fetchHistory = async () => {
                setIsLoadingHistory(true);
                try {
                    const endpoint = isLead ? `lead/binance-history?symbol=${symbol}` : `trades/history?symbol=${symbol}`;
                    const res = await fetch(`${API_BASE}/${endpoint}`);
                    if (res.ok) {
                        const data = await res.json();
                        setBinanceHistory(Array.isArray(data) ? data : []);
                    }
                } catch (e) {
                    console.error("Failed to fetch history:", e);
                } finally {
                    setIsLoadingHistory(false);
                }
            };
            fetchHistory();
        }
    }, [activeTab, symbol, isLead, tradingMode]);

    const displayOrders = useMemo(() => {
        try {
            const orders = Array.isArray(openOrders) ? openOrders : [];
            
            // Sync logic for Lead mode
            let syncedOrders = orders;
            if (isLead && leadPositions.length > 0) {
                syncedOrders = orders.map(o => {
                    // Sync Smart Trade Virtual Positions
                    if (o.type === 'POSITION') {
                        const livePos = leadPositions.find(p => p.symbol === o.symbol);
                        if (livePos) {
                            return { ...o, origQty: Math.abs(parseFloat(livePos.position_amt)), price: parseFloat(livePos.entry_price) };
                        }
                    }
                    return o;
                });
            }

            const history = Array.isArray(tradeHistory) ? tradeHistory : [];
            
            let result = [];
            if (activeTab === 'history') {
                const merged = [...history, ...binanceHistory];
                result = filterOrdersBySymbol ? merged.filter(o => o && o.symbol === symbol) : merged;
            }
            else if (activeTab === 'global') result = syncedOrders;
            else result = filterOrdersBySymbol ? syncedOrders.filter(o => o && o.symbol === symbol) : syncedOrders;
            
            return Array.isArray(result) ? result : [];
        } catch (e) {
            console.error("[BottomPanel] Memo Error:", e);
            return [];
        }
    }, [openOrders, binanceHistory, tradeHistory, symbol, filterOrdersBySymbol, activeTab, isLead, leadPositions]);

    const theme = isLead ? {
        bg: 'bg-orange-600',
        activeBtn: 'bg-orange-600',
        border: 'border-orange-500/20',
        accent: 'text-orange-400'
    } : {
        bg: 'bg-blue-600',
        activeBtn: 'bg-blue-600',
        border: 'border-blue-500/20',
        accent: 'text-blue-400'
    };

    return (
        <div className="bg-slate-950 h-full rounded-2xl border border-slate-800 shadow-xl flex flex-col overflow-hidden min-h-[200px]" style={{ border: '2px solid #1e293b' }}>
            {/* Header Tabs */}
            <div className="flex items-center px-4 border-b border-slate-800 bg-slate-900/30 shrink-0 h-12">
                <div className="flex gap-1">
                    <button 
                        onClick={() => setActiveTab('orders')} 
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'orders' ? `${theme.activeBtn} text-white shadow-lg` : 'text-slate-500 hover:text-white'}`}
                    >
                        <ListFilter size={12} /> Symbol
                    </button>
                    <button 
                        onClick={() => setActiveTab('global')} 
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'global' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-white'}`}
                    >
                        <LayoutDashboard size={12} /> Global
                    </button>
                    <button 
                        onClick={() => setActiveTab('history')} 
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'history' ? 'bg-slate-800 text-white shadow-lg' : 'text-slate-500 hover:text-white'}`}
                    >
                        <History size={12} /> History
                    </button>
                </div>
                <div className="flex-grow"></div>
                {activeTab === 'orders' && (
                    <div className="flex items-center gap-2 px-2">
                        <span className="text-[8px] text-slate-600 font-black uppercase tracking-widest">Filter Symbol</span>
                        <input 
                            type="checkbox" 
                            checked={filterOrdersBySymbol} 
                            onChange={() => setFilterOrdersBySymbol(!filterOrdersBySymbol)} 
                            className={`w-3 h-3 accent-${isLead ? 'orange' : 'blue'}-500 cursor-pointer`}
                        />
                    </div>
                )}
            </div>

            {/* Table Area */}
            <div className="flex-1 overflow-y-auto custom-scrollbar bg-slate-950 relative">
                {isLoadingHistory && activeTab === 'history' && (
                    <div className="absolute inset-0 bg-slate-950/50 backdrop-blur-sm z-20 flex items-center justify-center">
                        <div className={`w-6 h-6 border-2 border-slate-800 border-t-${isLead ? 'orange' : 'blue'}-500 rounded-full animate-spin`}></div>
                    </div>
                )}
                <div className="min-w-full">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-slate-900 z-10 shadow-sm">
                            <tr className="text-slate-500 text-[9px] uppercase font-black tracking-widest border-b border-slate-800">
                                <th className="p-3 text-white pl-4">Pair</th>
                                <th className="p-3">Side</th>
                                <th className="p-3 text-center">Price</th>
                                <th className="p-3 text-center">Qty</th>
                                <th className="p-3 text-center">Status</th>
                                <th className="p-3 text-right pr-4">{activeTab === 'history' ? 'Time' : 'Action'}</th>
                            </tr>
                        </thead>
                        <tbody className="font-mono text-[10px]">
                            {displayOrders.length > 0 ? displayOrders.map((o, idx) => {
                                if (!o) return null;
                                const isHistory = activeTab === 'history';
                                
                                // Normalize fields based on Spot Order vs Futures Trade
                                const s = o.symbol || 'N/A';
                                const side = (o.side || (parseFloat(o.qty) > 0 ? 'BUY' : 'SELL')).toUpperCase();
                                
                                // Status handling
                                let status = o.status || 'NEW';
                                if (isHistory && !o.status) status = 'FILLED'; // Trades are always filled
                                
                                // Quantity handling (Spot uses origQty, Futures uses qty)
                                // If Close Position order (qty=0), show label
                                let q = o.origQty || o.qty || '0';
                                const isClosePos = isLead && parseFloat(q) === 0 && !isHistory;
                                const displayQty = isClosePos ? 'CLOSE ALL' : Math.abs(parseFloat(q));
                                
                                // Price handling (Spot uses price, Futures uses price)
                                // If Market/Stop Market (price=0), use stopPrice
                                let p = parseFloat(o.price || o.avgPrice || '0');
                                if (p === 0 && (o.stopPrice || o.activatePrice)) {
                                    p = parseFloat(o.stopPrice || o.activatePrice || '0');
                                }
                                
                                const cid = o.clientOrderId || '';
                                const lcid = o.listClientOrderId || '';
                                const isSmart = cid.startsWith('SMART_') || lcid.startsWith('LIST_SMART_') || cid.startsWith('LEAD_');
                                
                                const pnl = o.realizedPnl ? parseFloat(o.realizedPnl) : null;
                                
                                return (
                                    <tr key={`${o.orderId || o.id || idx}-${idx}`} className={`border-b border-slate-800/20 hover:bg-slate-800/40 transition-colors ${isSmart ? `bg-${isLead ? 'orange' : 'blue'}-500/5` : ''}`}>
                                        <td className="p-3 pl-4 font-black text-slate-200">
                                            <div className="flex items-center gap-2">
                                                {s}
                                                {isSmart && (
                                                    <span className={`text-[7px] ${isLead ? 'bg-orange-600' : 'bg-blue-600'} text-white px-1 py-0.5 rounded font-black tracking-tighter uppercase`}>
                                                        {isLead ? 'LEAD' : 'SMART'}
                                                    </span>
                                                )}
                                            </div>
                                        </td>
                                        <td className={`p-3 font-black ${side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {side}
                                        </td>
                                        <td className="p-3 text-center text-slate-400">
                                            ${p.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                        </td>
                                        <td className="p-3 text-center text-slate-500">{displayQty}</td>
                                        <td className="p-3 text-center">
                                            {isHistory && pnl !== null ? (
                                                <span className={`font-black ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                                    {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                                                </span>
                                            ) : (
                                                <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold ${
                                                    status === 'FILLED' ? 'bg-emerald-900/30 text-emerald-500' : 
                                                    status === 'CANCELED' || status === 'REJECTED' ? 'bg-rose-900/30 text-rose-500' :
                                                    'bg-slate-800 text-slate-500'
                                                }`}>{status}</span>
                                            )}
                                        </td>
                                        <td className="p-3 text-right pr-4">
                                            {isHistory ? (
                                                <span className="text-slate-600 text-[9px] whitespace-nowrap">
                                                    {new Date(o.updateTime || o.time || Date.now()).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                                                </span>
                                            ) : (
                                                <button 
                                                    onClick={() => handleCancelOrder(o.orderId, s)} 
                                                    className="text-rose-500 hover:text-rose-400 text-[9px] font-black uppercase tracking-widest border border-rose-500/20 px-2 py-1 rounded-lg hover:border-rose-500/50 transition-all"
                                                >
                                                    Cancel
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                );
                            }) : (
                                <tr>
                                    <td colSpan={6} className="py-20 text-center text-slate-700 text-[10px] font-black uppercase tracking-[0.3em] opacity-20 bg-slate-950/50">
                                        No {activeTab} Records Found
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default BottomPanel;
