import { useMemo, useState, useEffect } from 'react';
import { LayoutDashboard, ShieldAlert, XCircle, ExternalLink, Activity, TrendingUp, TrendingDown, Shield, Zap, AlertTriangle, History as HistoryIcon, ArrowUpRight, ArrowDownRight, Wallet, Target as TargetIcon } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const WalletBalances = ({ balances, quoteAsset, globalPrices = [] }) => {
  if (!balances) return null;
  const quoteBalance = balances.find(b => b.asset === quoteAsset);
  
  const otherBalances = useMemo(() => {
    return balances
      .filter(b => b.asset !== quoteAsset && (parseFloat(b.free) > 0 || parseFloat(b.locked) > 0))
      .map(b => {
        const priceObj = globalPrices.find(p => (p as any).symbol === `${b.asset}${quoteAsset}`);
        const price = priceObj ? parseFloat((priceObj as any).price) : 0;
        const total = parseFloat(b.free) + parseFloat(b.locked);
        return { ...b, price, value: total * price };
      })
      .sort((a, b) => b.value - a.value);
  }, [balances, globalPrices, quoteAsset]);

  const totalPortfolioValue = useMemo(() => {
    const quoteAmt = quoteBalance ? parseFloat(quoteBalance.free) + parseFloat(quoteBalance.locked) : 0;
    const othersValue = otherBalances.reduce((sum, b) => sum + b.value, 0);
    return quoteAmt + othersValue;
  }, [quoteBalance, otherBalances]);

  return (
    <div className="relative group">
      <div className="bg-slate-950 px-8 py-4 rounded-2xl border border-slate-800 flex flex-col items-end shadow-[0_0_40px_rgba(0,0,0,0.5)] transition-all group-hover:border-blue-500 group-hover:shadow-blue-900/20 cursor-help min-w-[200px]">
        <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-1 flex items-center gap-2">
          <Wallet size={12} className="text-blue-400" /> Wallet Value
        </span>
        <p className="font-mono text-3xl font-black text-white tracking-tighter">
          ${totalPortfolioValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </p>
      </div>

      {/* MEGA DROP DOWN */}
      <div className="absolute top-full right-0 mt-4 w-[480px] bg-slate-900/98 backdrop-blur-2xl border border-slate-700/50 rounded-3xl shadow-[0_30px_100px_rgba(0,0,0,0.9)] overflow-hidden opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[100] transform origin-top-right group-hover:translate-y-0 translate-y-4 scale-95 group-hover:scale-100">
        <div className="p-6 border-b border-slate-800 bg-slate-950/50 flex justify-between items-center text-left">
          <h4 className="text-xs font-black text-white uppercase tracking-[0.2em] flex items-center gap-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            Portfolio Breakdown
          </h4>
          <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-800 px-3 py-1 rounded-full border border-slate-700">Spot Wallet</span>
        </div>
        
        <div className="p-2">
          <table className="w-full text-left border-separate border-spacing-y-1">
            <thead>
              <tr className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                <th className="px-4 py-3">Asset</th>
                <th className="px-4 py-3 text-right">Balance</th>
                <th className="px-4 py-3 text-right text-blue-400/70">Price</th>
                <th className="px-4 py-3 text-right text-emerald-400">Value ({quoteAsset})</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {/* Quote Asset Row First */}
              <tr className="bg-blue-500/5 rounded-xl transition-colors">
                <td className="px-4 py-4 text-sm text-white font-black">{quoteAsset}</td>
                <td className="px-4 py-4 text-right text-sm text-slate-300">{(parseFloat(quoteBalance?.free || '0') + parseFloat(quoteBalance?.locked || '0')).toLocaleString()}</td>
                <td className="px-4 py-4 text-right text-sm text-slate-600">1.00</td>
                <td className="px-4 py-4 text-right text-sm text-emerald-400 font-black">${parseFloat(quoteBalance?.free || '0').toLocaleString()}</td>
              </tr>
              
              {otherBalances.map(b => (
                <tr key={b.asset} className="hover:bg-slate-800/50 transition-colors">
                  <td className="px-4 py-4 text-sm text-white font-black">{b.asset}</td>
                  <td className="px-4 py-4 text-right text-sm text-slate-400">{parseFloat(b.free).toFixed(4)}</td>
                  <td className="px-4 py-4 text-right text-sm text-slate-500">${b.price < 1 ? b.price.toFixed(6) : b.price.toFixed(2)}</td>
                  <td className="px-4 py-4 text-right text-sm text-white font-black">${b.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="bg-slate-950/80 p-4 border-t border-slate-800 flex justify-between items-center text-left">
           <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Live exchange rates from Binance</span>
           <div className="flex gap-4">
              <div className="flex flex-col items-end">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">Cash</span>
                <span className="text-xs font-black text-slate-300">${parseFloat(quoteBalance?.free || '0').toLocaleString()}</span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">In Assets</span>
                <span className="text-xs font-black text-blue-400">${otherBalances.reduce((s, b) => s + b.value, 0).toLocaleString()}</span>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};

const SmartTradeCard = ({ trade, onCancel, currentPrice, onSelectSymbol }) => {
    const [isClosing, setIsClosing] = useState(false);
    const navigate = useNavigate();

    const handleSymbolClick = () => {
        onSelectSymbol(trade.symbol);
        navigate('/');
    };
    
    const entryPrice = trade.smart_meta?.entry_price || parseFloat(trade.price) || 0;
    const tpPrice = trade.smart_meta?.tp || 0;
    const slPrice = trade.smart_meta?.sl || 0;
    const tradeSide = trade.smart_meta?.side || trade.side;
    const isSyncing = !currentPrice || currentPrice === 0;
    const pnl = isSyncing ? 0 : ((currentPrice - entryPrice) / entryPrice * 100 * (tradeSide === 'BUY' ? 1 : -1));
    
    const getPos = (price) => {
        if (!price || !tpPrice || !slPrice) return 50;
        const totalRange = tpPrice - slPrice;
        if (totalRange === 0) return 50;
        const currentPos = price - slPrice;
        return Math.min(Math.max((currentPos / totalRange) * 100, 0), 100);
    };

    const entryMarkerPos = getPos(entryPrice);
    const currentMarkerPos = isSyncing ? entryMarkerPos : getPos(currentPrice);

    // Format for high precision
    const fmt = (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: val < 1 ? 6 : 2, maximumFractionDigits: val < 1 ? 8 : 2 });

    return (
        <div className="bg-slate-950 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden hover:border-blue-500/30 transition-all group w-full mb-4">
            <div className="flex flex-col xl:flex-row">
                {/* LEFT SECTION: INFO */}
                <div className="p-5 flex flex-col justify-center border-b xl:border-b-0 xl:border-r border-slate-800 min-w-[220px] bg-slate-900/20">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 bg-blue-600/10 rounded-lg">
                            <Activity size={18} className="text-blue-400" />
                        </div>
                        <button 
                            onClick={handleSymbolClick}
                            className="text-lg font-black text-white tracking-tighter uppercase hover:text-blue-400 transition-colors hover:underline"
                        >
                            {trade.symbol}
                        </button>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className={`text-[9px] font-black px-2 py-0.5 rounded ${tradeSide === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                            {tradeSide === 'BUY' ? 'LONG' : 'SHORT'}
                        </span>
                        <span className="text-[9px] text-slate-600 font-bold font-mono">{trade.clientOrderId}</span>
                    </div>
                </div>

                {/* MIDDLE SECTION: THE ACCURATE BAR */}
                <div className="flex-1 p-6 flex flex-col justify-center space-y-8">
                    <div className="relative pt-2">
                        {/* Background Track */}
                        <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800/50 relative">
                            <div 
                                className={`absolute h-full transition-all duration-1000 ${pnl >= 0 ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]'}`}
                                style={{ 
                                    left: `${Math.min(entryMarkerPos, currentMarkerPos)}%`, 
                                    width: `${Math.abs(currentMarkerPos - entryMarkerPos)}%` 
                                }}
                            ></div>
                        </div>
                        
                        {/* Legend Markers */}
                        <div className="absolute top-[-12px] w-full text-[8px] font-black uppercase tracking-tighter text-slate-600">
                            <span className="absolute left-0 text-rose-500">Stop Loss</span>
                            <span className="absolute" style={{ left: `${entryMarkerPos}%`, transform: 'translateX(-50%)' }}>Entry</span>
                            <span className="absolute right-0 text-emerald-500">Take Profit</span>
                        </div>

                        {/* Current Price Indicator */}
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

                    {/* Target Data Grid (Replaces overlapping numbers) */}
                    <div className="grid grid-cols-3 gap-6 text-center">
                        <div className="flex flex-col">
                            <span className="text-[10px] font-black text-rose-500/50 uppercase tracking-widest mb-1">SL Level</span>
                            <span className="text-sm font-mono font-bold text-slate-400">${fmt(slPrice)}</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Avg Entry</span>
                            <span className="text-sm font-mono font-bold text-slate-200">${fmt(entryPrice)}</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-[10px] font-black text-emerald-500/50 uppercase tracking-widest mb-1">TP Target</span>
                            <span className="text-sm font-mono font-bold text-slate-400">${fmt(tpPrice)}</span>
                        </div>
                    </div>
                </div>

                {/* RIGHT SECTION: P&L + BUTTONS */}
                <div className="p-5 flex flex-col xl:flex-row items-center gap-6 border-t xl:border-t-0 xl:border-l border-slate-800 bg-slate-900/10 min-w-[320px]">
                    <div className="text-right flex-1 w-full xl:w-auto">
                        {isSyncing ? (
                            <div className="flex flex-col items-end gap-1 animate-pulse">
                                <div className="h-8 w-24 bg-slate-800 rounded-lg mb-1"></div>
                                <div className="h-4 w-32 bg-slate-800/50 rounded-lg"></div>
                            </div>
                        ) : (
                            <>
                                <p className={`text-3xl font-black font-mono tracking-tighter ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                                </p>
                                <div className="flex items-center justify-end gap-2 mt-1">
                                    <span className="text-xs text-slate-500 font-black uppercase tracking-widest">Profit:</span>
                                    <span className={`text-lg font-mono font-black ${pnl >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                        ${((Math.abs(currentPrice - entryPrice) * trade.origQty) * (pnl >= 0 ? 1 : -1)).toFixed(2)}
                                    </span>
                                </div>
                            </>
                        )}
                    </div>

                    <div className="flex gap-2 w-full xl:w-auto">
                        <button className="p-3 bg-slate-800 hover:bg-slate-700 text-blue-400 rounded-xl transition-all" title="Break-Even">
                            <Zap size={16} />
                        </button>
                        <button 
                            onClick={async () => { setIsClosing(true); await onCancel(trade.orderId, trade.symbol, trade); setIsClosing(false); }}
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

const SmartHistoryTable = ({ history, onSelectSymbol }) => {
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
                        <th className="p-6 text-center">Net P&L</th>
                        <th className="p-6 text-center">Opened At</th>
                        <th className="p-6 text-center">Closed At</th>
                        <th className="p-6 text-center">Duration</th>
                    </tr>
                </thead>
                <tbody className="font-mono text-xs text-slate-300">
                    {history.length > 0 ? history.map(h => {
                        const grossPnl = ((h.exit_price - h.entry_price) / h.entry_price * 100 * (h.side === 'BUY' ? 1 : -1));
                        const entryFeeValue = h.fee_asset === h.symbol.replace('USDT','').replace('USDC','') ? (h.entry_fees * h.entry_price) : h.entry_fees;
                        const exitFeeValue = h.exit_fees;
                        const totalFees = (entryFeeValue || 0) + (exitFeeValue || 0);
                        const invested = h.quantity * h.entry_price;
                        const grossProfit = invested * (grossPnl / 100);
                        const netProfit = grossProfit - totalFees;
                        const netPnlPercent = (netProfit / invested) * 100;

                        return (
                            <tr key={h.id} className="border-b border-slate-800/30 hover:bg-slate-800/10 transition-all">
                                <td className="p-6">
                                    <button 
                                        onClick={() => { onSelectSymbol(h.symbol); navigate('/'); }}
                                        className="font-black text-white hover:text-blue-400 transition-colors hover:underline text-left outline-none"
                                    >
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
                        <tr><td colSpan={8} className="py-32 text-center text-slate-700 uppercase font-black opacity-30 tracking-widest">No Closed Trades Yet</td></tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}

const ActivePositionsView = ({ openOrders, handleCancelOrder, quoteBalance, onSelectSymbol, balances, globalPrices }) => {
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
            const qty = trade?.origQty || trade?.smart_meta?.quantity || 0;
            if (window.confirm(`Do you want to Market Sell the ${qty} ${symbol} associated with this Smart Trade?`)) {
                try {
                    const res = await fetch(`${API_BASE}/trades/market-close`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ symbol, quantity: qty, orderListId: trade?.smart_meta?.orderListId, clientOrderId: trade?.clientOrderId })
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
                </div>
                <div className="h-px w-full bg-slate-800"></div>
            </header>

            <div className="px-6 py-6 max-w-[1400px]">
                {activeTab === 'active' ? (
                    smartTrades.length > 0 ? (
                        <div className="flex flex-col gap-4">
                            {smartTrades.map(trade => (
                                <SmartTradeCard key={trade.orderId} trade={trade} onCancel={onCancelClick} currentPrice={prices[trade.symbol] || 0} onSelectSymbol={onSelectSymbol} />
                            ))}
                        </div>
                    ) : (
                        <div className="py-32 text-center bg-slate-950 rounded-3xl border border-slate-800 border-dashed opacity-20">
                            <Zap size={64} className="mx-auto mb-4" />
                            <p className="font-black text-xl tracking-widest uppercase">No Active Smart Trades</p>
                        </div>
                    )
                ) : (
                    <SmartHistoryTable history={smartHistory} onSelectSymbol={onSelectSymbol} />
                )}
            </div>
        </div>
    );
};

export default ActivePositionsView;
