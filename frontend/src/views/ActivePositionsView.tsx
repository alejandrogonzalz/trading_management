import { useMemo } from 'react';
import { LayoutDashboard, ShieldAlert, XCircle, ExternalLink, Activity } from 'lucide-react';

const ActivePositionsView = ({ openOrders, handleCancelOrder, usdcBalance }) => {
    
    const usdcOrders = useMemo(() => 
        openOrders.filter(o => o.symbol.endsWith('USDC')),
        [openOrders]
    );

    const totalCommitted = useMemo(() => 
        usdcOrders.reduce((sum, o) => sum + (Number(o.price) * Number(o.origQty)), 0),
        [usdcOrders]
    );

    return (
        <div className="h-full flex flex-col bg-slate-900 overflow-hidden">
            <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center px-6 py-6 gap-6">
                <div>
                    <h1 className="text-3xl font-black text-white flex items-center gap-3 tracking-tighter">
                        <LayoutDashboard size={32} className="text-indigo-500" /> 
                        ACTIVE POSITIONS
                    </h1>
                    <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest mt-1">Global USDC Portfolio Management</p>
                </div>

                <div className="flex gap-4">
                    <div className="bg-slate-950 px-6 py-3 rounded-2xl border border-slate-800 flex flex-col items-end shadow-lg">
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1 opacity-60">Available Cash</span>
                        <p className="font-mono text-xl font-black text-white tracking-tighter">${parseFloat(usdcBalance).toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                    </div>
                    <div className="bg-indigo-600/5 px-6 py-3 rounded-2xl border border-indigo-500/20 flex flex-col items-end shadow-lg">
                        <span className="text-[9px] font-black text-indigo-400 uppercase tracking-widest mb-1 opacity-70">Locked in Orders</span>
                        <p className="font-mono text-xl font-black text-indigo-100 tracking-tighter">${totalCommitted.toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                    </div>
                </div>
            </header>

            <div className="flex-1 min-h-0 bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden mx-6 mb-6 flex flex-col">
                <div className="bg-slate-900/30 px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Activity size={16} className="text-indigo-400" />
                        <span className="text-[10px] text-slate-400 font-black uppercase tracking-[0.15em]">{usdcOrders.length} Pending Smart Orders</span>
                    </div>
                </div>

                <div className="flex-1 overflow-auto scrollbar-thin scrollbar-thumb-slate-800">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-slate-900 z-10">
                            <tr className="text-slate-500 text-[10px] uppercase font-black tracking-widest border-b border-slate-800">
                                <th className="p-6">Symbol</th>
                                <th className="p-6 text-center">Type</th>
                                <th className="p-6 text-center">Side</th>
                                <th className="p-6 text-center">Price</th>
                                <th className="p-6 text-center">Quantity</th>
                                <th className="p-6 text-center">Total (USDC)</th>
                                <th className="p-6 text-center">Status</th>
                                <th className="p-6 text-right">Emergency</th>
                            </tr>
                        </thead>
                        <tbody className="font-mono text-sm">
                            {usdcOrders.length > 0 ? usdcOrders.map(o => (
                                <tr key={o.orderId} className="border-b border-slate-800/30 hover:bg-indigo-500/5 transition-all group">
                                    <td className="p-6">
                                        <div className="flex items-center gap-2">
                                            <span className="font-black text-white text-base">{o.symbol}</span>
                                            <ExternalLink size={12} className="text-slate-700 group-hover:text-blue-400 transition-colors cursor-pointer" />
                                        </div>
                                    </td>
                                    <td className="p-6 text-center text-slate-500 font-bold uppercase text-[10px]">{o.type}</td>
                                    <td className="p-6 text-center">
                                        <span className={`px-3 py-1 rounded-lg font-black text-[10px] ${o.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'}`}>
                                            {o.side}
                                        </span>
                                    </td>
                                    <td className="p-6 text-center text-slate-300 font-bold">${Number(o.price).toLocaleString()}</td>
                                    <td className="p-6 text-center text-slate-400">{o.origQty}</td>
                                    <td className="p-6 text-center text-white font-black">${(Number(o.price) * Number(o.origQty)).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                    <td className="p-6 text-center">
                                        <div className="flex items-center justify-center gap-2">
                                            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></div>
                                            <span className="text-[10px] font-black text-slate-500 uppercase">{o.status}</span>
                                        </div>
                                    </td>
                                    <td className="p-6 text-right">
                                        <button 
                                            onClick={() => handleCancelOrder(o.orderId, o.symbol)}
                                            className="flex items-center gap-2 ml-auto bg-slate-900 hover:bg-rose-900/20 text-slate-500 hover:text-rose-400 px-4 py-2 rounded-xl border border-slate-800 hover:border-rose-500/30 transition-all text-[10px] font-black uppercase tracking-widest"
                                        >
                                            <XCircle size={14} />
                                            Cancel
                                        </button>
                                    </td>
                                </tr>
                            )) : (
                                <tr>
                                    <td colSpan={8} className="py-32 text-center">
                                        <div className="flex flex-col items-center gap-4 opacity-20">
                                            <ShieldAlert size={64} />
                                            <p className="font-black text-xl tracking-widest uppercase text-white">No Active Positions</p>
                                            <p className="text-xs font-bold">Your USDC trading account has no pending orders.</p>
                                        </div>
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

export default ActivePositionsView;
