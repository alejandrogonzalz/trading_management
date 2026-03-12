import { useMemo } from 'react';
import { Activity, XCircle, ShieldAlert } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const ActiveOrdersView = ({ openOrders, handleCancelOrder, quoteBalance, onSelectSymbol }) => {
    const navigate = useNavigate();
    const relevantOrders = useMemo(() => 
        openOrders.filter(o => 
            !o.clientOrderId?.startsWith('SMART_') && 
            !o.listClientOrderId?.startsWith('LIST_SMART_') &&
            (o.symbol.endsWith('USDC') || o.symbol.endsWith('USDT'))
        ),
        [openOrders]
    );

    const onCancelClick = (orderId, symbol) => {
        if (window.confirm(`Are you sure you want to cancel order #${orderId} for ${symbol}?`)) {
            handleCancelOrder(orderId, symbol);
        }
    };

    return (
        <div className="h-full flex flex-col bg-slate-900 overflow-hidden">
            <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center px-6 py-6 gap-6">
                <div>
                    <h1 className="text-3xl font-black text-white flex items-center gap-3 tracking-tighter">
                        <Activity size={32} className="text-blue-500" /> 
                        ACTIVE ORDERS
                    </h1>
                    <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest mt-1">Bot & System Managed Orders</p>
                </div>
            </header>

            <div className="flex-1 min-h-0 bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden mx-6 mb-6 flex flex-col">
                <div className="flex-1 overflow-auto scrollbar-thin scrollbar-thumb-slate-800">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-slate-900 z-10">
                            <tr className="text-slate-500 text-[10px] uppercase font-black tracking-widest border-b border-slate-800">
                                <th className="p-6">Symbol</th>
                                <th className="p-6 text-center">Side</th>
                                <th className="p-6 text-center">Type</th>
                                <th className="p-6 text-center">Price</th>
                                <th className="p-6 text-center">Quantity</th>
                                <th className="p-6 text-center">Status</th>
                                <th className="p-6 text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="font-mono text-sm">
                            {relevantOrders.length > 0 ? relevantOrders.map(o => (
                                <tr key={o.orderId} className="border-b border-slate-800/30 hover:bg-slate-800/10 transition-all">
                                    <td className="p-6">
                                        <button 
                                            onClick={() => { onSelectSymbol(o.symbol); navigate('/'); }}
                                            className="font-black text-white hover:text-blue-400 transition-colors hover:underline text-left outline-none"
                                        >
                                            {o.symbol}
                                        </button>
                                    </td>
                                    <td className="p-6 text-center">
                                        <span className={`px-3 py-1 rounded-lg font-black text-[10px] ${o.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                            {o.side}
                                        </span>
                                    </td>
                                    <td className="p-6 text-center text-slate-500 font-bold uppercase text-[10px]">{o.type}</td>
                                    <td className="p-6 text-center text-slate-300 font-bold">${Number(o.price || 0).toLocaleString()}</td>
                                    <td className="p-6 text-center text-slate-400">{o.origQty}</td>
                                    <td className="p-6 text-center text-[10px] text-slate-500 font-black uppercase">{o.status}</td>
                                    <td className="p-6 text-right">
                                        <button onClick={() => onCancelClick(o.orderId, o.symbol)} className="text-rose-500 hover:text-white p-2 hover:bg-rose-600 rounded-xl transition-all">
                                            <XCircle size={18} />
                                        </button>
                                    </td>
                                </tr>
                            )) : (
                                <tr>
                                    <td colSpan={7} className="py-32 text-center">
                                        <div className="flex flex-col items-center gap-4 opacity-20">
                                            <ShieldAlert size={64} />
                                            <p className="font-black text-xl tracking-widest uppercase text-white">No Active Bot Orders</p>
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

export default ActiveOrdersView;
