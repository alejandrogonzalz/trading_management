import { useMemo } from 'react';
import { Activity, XCircle, ShieldAlert, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const ActiveOrdersView = ({ openOrders, handleCancelOrder, quoteBalance, onSelectSymbol, tradingMode, leadPositions = [] }) => {
    const navigate = useNavigate();
    const isLead = tradingMode === 'LEAD';

    const relevantOrders = useMemo(() => {
        if (isLead) {
            // For Lead/Futures, show real pending orders (SL/TP/Limit)
            // Filter out FILLED/CANCELED orders
            // Filter out virtual 'POSITION' markers (which are just for the Smart Terminal view)
            const raw = openOrders.filter(o => 
                o.status !== 'FILLED' && 
                o.status !== 'CANCELED' && 
                o.type !== 'POSITION'
            );

            // Sync Smart Trades (which appear as orders) with live position data
            return raw.map(o => {
                if (o.clientOrderId?.startsWith('LEAD_') || o.orderId?.toString().startsWith('LEAD_')) {
                    const livePos = leadPositions.find(p => p.symbol === o.symbol);
                    if (livePos) {
                        return { 
                            ...o, 
                            origQty: Math.abs(parseFloat(livePos.position_amt)),
                            price: parseFloat(livePos.entry_price) || o.price
                        };
                    }
                }
                return o;
            });
        }
        // For Spot, maintain the "Protector Leg" filter
        return openOrders.filter(o => o.clientOrderId?.startsWith('SMART_') || o.listClientOrderId?.startsWith('LIST_SMART_'));
    }, [openOrders, isLead, leadPositions]);

    const theme = isLead ? {
        text: 'text-orange-500',
        activeText: 'text-orange-400',
        border: 'border-orange-500',
        btn: 'hover:text-orange-400'
    } : {
        text: 'text-blue-500',
        activeText: 'text-blue-400',
        border: 'border-blue-500',
        btn: 'hover:text-blue-400'
    };

    const formatQuantity = (qty) => {
        if (typeof qty === 'string' && qty === 'CLOSE ALL') return qty;
        const num = parseFloat(qty);
        if (num === 0) return 'CLOSE ALL';
        const dynamicPrecision = (q) => {
            if (q >= 1000) return 0;
            if (q >= 1) return 2;
            if (q >= 0.001) return 6;
            return 8;
        };
        const precision = dynamicPrecision(num);
        return num.toLocaleString(undefined, { 
            minimumFractionDigits: precision, 
            maximumFractionDigits: precision 
        });
    };

    const formatPrice = (price) => {
        const dynamicPrecision = (p) => {
            if (p < 0.001) return 8;
            if (p < 0.1) return 6;
            if (p < 1) return 4;
            return 2;
        };
        return price.toLocaleString(undefined, { 
            minimumFractionDigits: dynamicPrecision(price), 
            maximumFractionDigits: dynamicPrecision(price) 
        });
    };

    return (
        <div className="h-full flex flex-col bg-slate-900 overflow-hidden">
            <header className="px-6 pt-8 pb-4 shrink-0">
                <div className="flex justify-between items-end">
                    <div>
                        <h1 className="text-3xl font-black text-white flex items-center gap-3 tracking-tighter uppercase">
                            <ShieldAlert size={32} className={theme.text} /> {isLead ? 'FUTURES ORDERS' : 'SYSTEM ORDERS'}
                        </h1>
                        <p className="text-slate-500 text-[10px] font-black uppercase tracking-[0.2em] mt-2">
                            {isLead ? 'Active USDS-M Futures Orders' : 'Active Protector Legs & Smart Setups'}
                        </p>
                    </div>
                    <div className="bg-slate-950 px-6 py-3 rounded-2xl border border-slate-800 flex flex-col items-end shadow-lg mb-2">
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1 opacity-60">
                            {isLead ? 'Available Margin' : 'Total Cash'}
                        </span>
                        <p className="font-mono text-xl font-black text-white tracking-tighter">${parseFloat(quoteBalance).toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                    </div>
                </div>
                <div className="h-px w-full bg-slate-800 mt-4"></div>
            </header>

            <div className="flex-1 overflow-y-auto px-6 py-6 custom-scrollbar">
                <div className="bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden max-w-[1200px]">
                    <table className="w-full text-left border-collapse">
                        <thead className="bg-slate-900">
                            <tr className="text-slate-500 text-[10px] uppercase font-black tracking-widest border-b border-slate-800">
                                <th className="p-6">Symbol</th>
                                <th className="p-6 text-center">Type</th>
                                <th className="p-4 text-center">Side</th>
                                <th className="p-6 text-center">Price</th>
                                <th className="p-6 text-center">Quantity</th>
                                {isLead && <th className="p-6 text-center">Leverage</th>}
                                <th className="p-6 text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="font-mono text-xs text-slate-300">
                            {relevantOrders.length > 0 ? relevantOrders.map(o => (
                                <tr key={o.orderId} className="border-b border-slate-800/30 hover:bg-slate-800/10 transition-all">
                                    <td className="p-6">
                                        <button 
                                            onClick={() => { onSelectSymbol(o.symbol); navigate('/'); }}
                                            className={`font-black text-white ${theme.btn} transition-colors hover:underline text-left outline-none`}
                                        >
                                            {o.symbol}
                                        </button>
                                    </td>
                                    <td className="p-6 text-center text-slate-500 font-bold uppercase tracking-widest text-[10px]">{o.type}</td>
                                    <td className="p-4 text-center">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-black ${o.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                            {o.side}
                                        </span>
                                    </td>
                                    <td className="p-6 text-center text-white font-black">
                                        ${formatPrice(parseFloat(o.stopPrice || o.price || '0'))}
                                    </td>
                                    <td className="p-6 text-center text-slate-400">
                                        {formatQuantity(parseFloat(o.origQty || o.qty) === 0 ? 'CLOSE ALL' : (o.origQty || o.qty))}
                                    </td>
                                    {isLead && <td className="p-6 text-center text-orange-400">{o.leverage || '--'}x</td>}
                                    <td className="p-6 text-right">
                                        <button 
                                            onClick={() => handleCancelOrder(o.orderId, o.symbol)}
                                            className="px-4 py-1.5 rounded-lg bg-rose-600/10 hover:bg-rose-600 text-rose-500 hover:text-white text-[10px] font-black uppercase tracking-widest transition-all border border-rose-500/20"
                                        >
                                            Cancel Order
                                        </button>
                                    </td>
                                </tr>
                            )) : (
                                <tr>
                                    <td colSpan={isLead ? 7 : 6} className="py-32 text-center text-slate-700 uppercase font-black opacity-30 tracking-widest">
                                        <ShieldAlert size={48} className="mx-auto mb-4 opacity-20" />
                                        No {isLead ? 'Futures' : 'System'} Orders Active
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
