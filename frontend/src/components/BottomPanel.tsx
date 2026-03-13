import { useState, useMemo, useEffect } from 'react';
import { ListFilter, LayoutDashboard, History } from 'lucide-react';

const BottomPanel = ({ 
    openOrders = [], 
    tradeHistory = [], 
    symbol = '', 
    filterOrdersBySymbol = true, 
    setFilterOrdersBySymbol = () => {}, 
    handleCancelOrder = () => {} 
}) => {
    const [activeTab, setActiveTab] = useState('orders');

    const displayOrders = useMemo(() => {
        try {
            const orders = Array.isArray(openOrders) ? openOrders : [];
            const history = Array.isArray(tradeHistory) ? tradeHistory : [];
            
            let result = [];
            if (activeTab === 'history') result = history;
            else if (activeTab === 'global') result = orders;
            else result = filterOrdersBySymbol ? orders.filter(o => o && o.symbol === symbol) : orders;
            
            return Array.isArray(result) ? result : [];
        } catch (e) {
            console.error("[BottomPanel] Memo Error:", e);
            return [];
        }
    }, [openOrders, tradeHistory, symbol, filterOrdersBySymbol, activeTab]);

    return (
        <div className="bg-slate-950 h-full rounded-2xl border border-slate-800 shadow-xl flex flex-col overflow-hidden min-h-[200px]" style={{ border: '2px solid #1e293b' }}>
            {/* Header Tabs */}
            <div className="flex items-center px-4 border-b border-slate-800 bg-slate-900/30 shrink-0 h-12">
                <div className="flex gap-1">
                    <button 
                        onClick={() => setActiveTab('orders')} 
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'orders' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-white'}`}
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
                            className="w-3 h-3 accent-blue-500 cursor-pointer" 
                        />
                    </div>
                )}
            </div>

            {/* Table Area */}
            <div className="flex-1 overflow-y-auto custom-scrollbar bg-slate-950">
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
                                // Extremely safe property checks to prevent crashes
                                const s = o.symbol || 'N/A';
                                const side = o.side || 'N/A';
                                const status = o.status || 'N/A';
                                const q = o.origQty || '0';
                                const p = o.price || '0';
                                const cid = o.clientOrderId || '';
                                const lcid = o.listClientOrderId || '';
                                const isSmart = cid.startsWith('SMART_') || lcid.startsWith('LIST_SMART_');
                                
                                return (
                                    <tr key={`${o.orderId || idx}-${idx}`} className={`border-b border-slate-800/20 hover:bg-slate-800/40 transition-colors ${isSmart ? 'bg-blue-500/5' : ''}`}>
                                        <td className="p-3 pl-4 font-black text-slate-200">
                                            <div className="flex items-center gap-2">
                                                {s}
                                                {isSmart && (
                                                    <span className="text-[7px] bg-blue-600 text-white px-1 py-0.5 rounded font-black tracking-tighter">SMART</span>
                                                )}
                                            </div>
                                        </td>
                                        <td className={`p-3 font-black ${side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {side}
                                        </td>
                                        <td className="p-3 text-center text-slate-400">
                                            {(() => {
                                                try {
                                                    if (status === 'FILLED' && (!p || Number(p) === 0)) {
                                                        const qQty = Number(o.cummulativeQuoteQty || 0);
                                                        const eQty = Number(o.executedQty || 1);
                                                        return `$${(qQty / eQty).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
                                                    }
                                                    return `$${Number(p).toLocaleString()}`;
                                                } catch(err) { return '$0.00'; }
                                            })()}
                                        </td>
                                        <td className="p-3 text-center text-slate-500">{q}</td>
                                        <td className="p-3 text-center">
                                            <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold ${
                                                status === 'FILLED' ? 'bg-emerald-900/30 text-emerald-500' : 
                                                status === 'CANCELED' ? 'bg-rose-900/30 text-rose-500' :
                                                'bg-slate-800 text-slate-500'
                                            }`}>{status}</span>
                                        </td>
                                        <td className="p-3 text-right pr-4">
                                            {activeTab === 'history' ? (
                                                <span className="text-slate-600 text-[9px] whitespace-nowrap">
                                                    {o.updateTime ? new Date(o.updateTime).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : 'N/A'}
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
