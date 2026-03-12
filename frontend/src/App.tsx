import { useState, useEffect, useMemo } from 'react';
import TradingChart from './components/TradingChart';
import Sidebar from './components/Sidebar';
import { Route, Routes, HashRouter as BrowserRouter, useNavigate } from 'react-router-dom';
import ScannerView from './views/ScannerView';
import ActivePositionsView from './views/ActivePositionsView';
import ActiveOrdersView from './views/ActiveOrdersView';
import { ScanSearch, ListFilter, History, LayoutDashboard, ArrowUp, ArrowDown, Zap, Wallet } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const WalletBalances = ({ balances, quoteAsset, globalPrices }) => {
  const quoteBalance = balances.find(b => b.asset === quoteAsset);
  const otherBalances = useMemo(() => {
    return balances
      .filter(b => b.asset !== quoteAsset && (parseFloat(b.free) > 0 || parseFloat(b.locked) > 0))
      .map(b => {
        const priceObj = globalPrices.find(p => p.symbol === `${b.asset}${quoteAsset}`);
        const price = priceObj ? parseFloat(priceObj.price) : 0;
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
      <div className="bg-blue-600/5 px-6 py-2 rounded-xl border border-blue-500/20 flex flex-col items-end justify-center shadow-lg transition-all group-hover:border-blue-500/50 group-hover:bg-blue-600/10 cursor-help min-w-[160px] h-[52px]">
        <span className="text-[8px] font-black text-blue-400 uppercase tracking-[0.2em] mb-0.5 flex items-center gap-1">
          <Wallet size={10} /> Wallet Value
        </span>
        <p className="font-mono text-lg font-black text-blue-100 tracking-tighter">
          ${totalPortfolioValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </p>
      </div>

      {/* MEGA DROP DOWN */}
      <div className="absolute top-full right-0 mt-4 w-[480px] bg-slate-900/98 backdrop-blur-2xl border border-slate-700/50 rounded-3xl shadow-[0_30px_100px_rgba(0,0,0,0.9)] overflow-hidden opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[2000] transform origin-top-right group-hover:translate-y-0 translate-y-4 scale-95 group-hover:scale-100">
        <div className="p-6 border-b border-slate-800 bg-slate-950/50 flex justify-between items-center">
          <h4 className="text-xs font-black text-white uppercase tracking-[0.2em] flex items-center gap-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            Portfolio Breakdown
          </h4>
          <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-800 px-3 py-1 rounded-full border border-slate-700">Spot Wallet</span>
        </div>
        
        <div className="p-2">
          <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar relative">
            <table className="w-full text-left border-collapse">
              <thead className="sticky top-0 z-20">
                <tr className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-900">
                  <th className="px-4 py-3 border-b border-slate-800">Asset</th>
                  <th className="px-4 py-3 text-right border-b border-slate-800">Balance</th>
                  <th className="px-4 py-3 text-right text-blue-400/70 border-b border-slate-800">Price</th>
                  <th className="px-4 py-3 text-right text-emerald-400 border-b border-slate-800">Value ({quoteAsset})</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {/* Quote Asset Row First */}
                <tr className="bg-blue-500/10 transition-colors border-b border-slate-800/50">
                  <td className="px-4 py-4 text-sm text-white font-black">{quoteAsset}</td>
                  <td className="px-4 py-4 text-right text-sm text-slate-300">{(parseFloat(quoteBalance?.free || '0') + parseFloat(quoteBalance?.locked || '0')).toLocaleString()}</td>
                  <td className="px-4 py-4 text-right text-sm text-slate-600">1.00</td>
                  <td className="px-4 py-4 text-right text-sm text-emerald-400 font-black">${parseFloat(quoteBalance?.free || '0').toLocaleString()}</td>
                </tr>
                
                {otherBalances.map(b => (
                  <tr key={b.asset} className="hover:bg-slate-800/50 transition-colors border-b border-slate-800/30 last:border-0">
                    <td className="px-4 py-4 text-sm text-white font-black">{b.asset}</td>
                    <td className="px-4 py-4 text-right text-sm text-slate-400">{parseFloat(b.free).toFixed(4)}</td>
                    <td className="px-4 py-4 text-right text-sm text-slate-500">${b.price < 1 ? b.price.toFixed(6) : b.price.toFixed(2)}</td>
                    <td className="px-4 py-4 text-right text-sm text-white font-black">${b.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {otherBalances.length === 0 && (
            <div className="py-12 text-center">
              <Zap size={32} className="mx-auto text-slate-800 mb-2" />
              <p className="text-[10px] text-slate-600 font-bold uppercase tracking-widest">No other assets detected</p>
            </div>
          )}
        </div>
        
        <div className="bg-slate-950/80 p-4 border-t border-slate-800 flex justify-between items-center">
           <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Live rates from Binance</span>
           <div className="flex gap-4">
              <div className="flex flex-col items-end">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">Total {quoteAsset}</span>
                <span className="text-xs font-black text-slate-300">
                  ${(parseFloat(quoteBalance?.free || '0') + parseFloat(quoteBalance?.locked || '0')).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">In Assets</span>
                <span className="text-xs font-black text-blue-400">
                  ${otherBalances.reduce((s, b) => s + b.value, 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};

const SmartTerminalView = ({ 
  symbol, currentPrice, handleSmartTrade, handleMarketClose, 
  tpPrice, setTpPrice, slPrice, setSlPrice, 
  tpEnabled, setTpEnabled, slEnabled, setSlEnabled, 
  assetBalance, tradingMode, isTrading,
  tpPercent, setTpPercent, slPercent, setSlPercent
}) => {
  const [quantity, setQuantity] = useState(0.001);
  const [usdcAmount, setUsdcAmount] = useState(0);
  const [side, setSide] = useState('BUY'); // 'BUY' (Long) or 'SELL' (Short)

  // Force 'BUY' if in SPOT mode
  useEffect(() => {
    if (tradingMode === 'SPOT') setSide('BUY');
  }, [tradingMode]);

  // Sync USDC when quantity changes
  const handleQuantityChange = (val: number) => {
    setQuantity(val);
    if (currentPrice > 0) {
      setUsdcAmount(Number((val * currentPrice).toFixed(2)));
    }
  };

  // Sync Quantity when USDC changes
  const handleUsdcChange = (val: number) => {
    setUsdcAmount(val);
    if (currentPrice > 0) {
      setQuantity(Number((val / currentPrice).toFixed(6)));
    }
  };

  useEffect(() => {
    if (currentPrice > 0 && usdcAmount === 0) {
      setUsdcAmount(Number((quantity * currentPrice).toFixed(2)));
    }
  }, [currentPrice]);

  // Re-sync prices if currentPrice changes or if Percent is typed manually
  const handleTpPercentChange = (val: number) => {
    const fixedVal = Number(val.toFixed(4));
    setTpPercent(fixedVal);
    if (currentPrice > 0) {
      setTpPrice(Number((currentPrice * (1 + fixedVal / 100)).toFixed(8)));
    }
  };

  const handleSlPercentChange = (val: number) => {
    const fixedVal = Number(val.toFixed(4));
    setSlPercent(fixedVal);
    if (currentPrice > 0) {
      setSlPrice(Number((currentPrice * (1 + fixedVal / 100)).toFixed(8)));
    }
  };

  const handleTpPriceChange = (val: number) => {
    setTpPrice(val);
    if (currentPrice > 0) {
      setTpPercent(Number(((val - currentPrice) / currentPrice * 100).toFixed(4)));
    }
  };

  const handleSlPriceChange = (val: number) => {
    setSlPrice(val);
    if (currentPrice > 0) {
      setSlPercent(Number(((val - currentPrice) / currentPrice * 100).toFixed(4)));
    }
  };
  
  const onTradeClick = () => {
    handleSmartTrade({
      quantity,
      tpPrice: tpEnabled ? tpPrice : 0,
      slPrice: slEnabled ? slPrice : 0,
      side,
      mode: tradingMode
    });
  }

  const quoteAsset = symbol.endsWith('USDT') ? 'USDT' : 'USDC';
  const modeColor = tradingMode === 'LEAD' ? 'orange' : 'blue';

  return (
    <div className={`bg-slate-950 p-6 h-full rounded-2xl border ${tradingMode === 'LEAD' ? 'border-orange-900/30 shadow-orange-900/10' : 'border-slate-800'} shadow-xl overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 transition-all duration-500`}>
        <h2 className="text-xl font-black mb-6 tracking-tight flex items-center justify-between text-white uppercase">
            <div className="flex items-center gap-2">
                <div className={`w-2 h-2 bg-${modeColor}-500 rounded-full animate-pulse`}></div>
                {tradingMode === 'SPOT' ? 'Smart Buy Strategy' : 'Lead Trading Terminal'}
            </div>
            {tradingMode === 'LEAD' && <span className="text-[10px] bg-orange-500/10 text-orange-400 px-2 py-0.5 rounded border border-orange-500/20">FUTURES</span>}
        </h2>
      
      <div className="space-y-6">
        {/* Side Selector (Only for Lead Trading) */}
        {tradingMode === 'LEAD' && (
          <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800">
            <button 
              onClick={() => setSide('BUY')}
              className={`flex-1 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${side === 'BUY' ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/40' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Long Position
            </button>
            <button 
              onClick={() => setSide('SELL')}
              className={`flex-1 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${side === 'SELL' ? 'bg-rose-600 text-white shadow-lg shadow-rose-900/40' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Short Position
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4">
          <div className="space-y-2">
            <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest ml-1">Spend ({quoteAsset})</label>
            <input 
              type="number" value={usdcAmount} onChange={e => handleUsdcChange(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-lg font-mono outline-none focus:border-blue-500 transition-all text-white font-black"
            />
          </div>
          <div className="space-y-2">
            <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest ml-1">Receive ({symbol.replace(quoteAsset, '')})</label>
            <input 
              type="number" step="0.0001" value={quantity} onChange={e => handleQuantityChange(Number(e.target.value))}
              className="w-full bg-slate-800/30 border border-slate-800 rounded-xl px-4 py-3 text-lg font-mono outline-none focus:border-blue-500/50 transition-all text-slate-300"
            />
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-slate-800">
          <div className={`p-4 rounded-2xl border transition-all ${tpEnabled ? 'bg-emerald-900/10 border-emerald-900/30' : 'bg-slate-800/30 border-slate-800 opacity-40'}`}>
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-3">
                <input type="checkbox" checked={tpEnabled} onChange={() => setTpEnabled(!tpEnabled)} className="w-4 h-4 accent-emerald-500 cursor-pointer" />
                <label className="text-[10px] text-emerald-400 font-black uppercase tracking-widest">Take Profit</label>
              </div>
              {tpEnabled && (
                <div className="flex items-center gap-1 bg-slate-950 rounded-lg border border-emerald-500/20 px-1">
                  <button onClick={() => handleTpPercentChange(tpPercent - 0.1)} className="p-1 text-slate-600 hover:text-emerald-400"><ArrowDown size={12} /></button>
                  <input 
                    type="number" step="0.1" value={tpPercent} 
                    onChange={e => handleTpPercentChange(parseFloat(e.target.value))}
                    className="w-14 bg-transparent text-emerald-400 font-black text-xs outline-none text-right"
                  />
                  <span className="text-[10px] text-emerald-500 font-black pr-1">%</span>
                  <button onClick={() => handleTpPercentChange(tpPercent + 0.1)} className="p-1 text-slate-600 hover:text-emerald-400"><ArrowUp size={12} /></button>
                </div>
              )}
            </div>
            <input 
              type="number" value={tpPrice} onChange={e => handleTpPriceChange(Number(e.target.value))} disabled={!tpEnabled}
              className="w-full bg-transparent border-none text-2xl font-mono outline-none text-white font-black"
            />
          </div>

          <div className={`p-4 rounded-2xl border transition-all ${slEnabled ? 'bg-rose-900/10 border-rose-900/30' : 'bg-slate-800/30 border-slate-800 opacity-40'}`}>
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-3">
                <input type="checkbox" checked={slEnabled} onChange={() => setSlEnabled(!slEnabled)} className="w-4 h-4 accent-rose-500 cursor-pointer" />
                <label className="text-[10px] text-rose-400 font-black uppercase tracking-widest">Stop Loss</label>
              </div>
              {slEnabled && (
                <div className="flex items-center gap-1 bg-slate-950 rounded-lg border border-rose-500/20 px-1">
                  <button onClick={() => handleSlPercentChange(slPercent - 0.1)} className="p-1 text-slate-600 hover:text-rose-400"><ArrowDown size={12} /></button>
                  <input 
                    type="number" step="0.1" value={slPercent} 
                    onChange={e => handleSlPercentChange(parseFloat(e.target.value))}
                    className="w-14 bg-transparent text-rose-400 font-black text-xs outline-none text-right"
                  />
                  <span className="text-[10px] text-rose-500 font-black pr-1">%</span>
                  <button onClick={() => handleSlPercentChange(slPercent + 0.1)} className="p-1 text-slate-600 hover:text-rose-400"><ArrowUp size={12} /></button>
                </div>
              )}
            </div>
            <input 
              type="number" value={slPrice} onChange={e => handleSlPriceChange(Number(e.target.value))} disabled={!slEnabled}
              className="w-full bg-transparent border-none text-2xl font-mono outline-none text-white font-black"
            />
          </div>
        </div>

        <button 
          onClick={onTradeClick}
          disabled={isTrading}
          className={`w-full ${isTrading ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : (tradingMode === 'SPOT' ? 'bg-blue-600 hover:bg-blue-500' : (side === 'BUY' ? 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-900/20' : 'bg-rose-600 hover:bg-rose-500 shadow-rose-900/20'))} text-white font-black py-4 rounded-2xl mt-4 transition-all shadow-lg uppercase tracking-widest text-sm flex items-center justify-center gap-3`}
        >
          {isTrading ? (
            <>
              <div className="w-4 h-4 border-2 border-slate-600 border-t-blue-400 rounded-full animate-spin"></div>
              Processing...
            </>
          ) : (
            tradingMode === 'SPOT' ? 'Execute Smart Trade' : `Execute Lead ${side === 'BUY' ? 'Long' : 'Short'}`
          )}
        </button>

        <div className="pt-4 border-t border-slate-800">
            <button
                onClick={() => handleMarketClose()}
                className="w-full bg-slate-800 hover:bg-rose-900/50 hover:text-rose-400 text-slate-400 font-black py-3 rounded-xl transition-all text-[10px] uppercase tracking-widest border border-slate-700"
            >
                Emergency Market Close ({assetBalance.toFixed(4)})
            </button>
        </div>
      </div>
    </div>
  );
};


const BottomPanel = ({ openOrders, tradeHistory, symbol, filterOrdersBySymbol, setFilterOrdersBySymbol, handleCancelOrder }) => {
    const [activeTab, setActiveTab] = useState('orders');

    const displayOrders = useMemo(() => {
        if (activeTab === 'history') return tradeHistory;
        if (activeTab === 'global') return openOrders;
        return filterOrdersBySymbol ? openOrders.filter(o => o.symbol === symbol) : openOrders;
    }, [openOrders, tradeHistory, symbol, filterOrdersBySymbol, activeTab]);

    const OrderTable = () => (
        <div className="overflow-x-auto">
            <table className="w-full text-left">
                <thead>
                    <tr className="text-slate-500 text-[9px] uppercase font-black tracking-[0.2em] border-b border-slate-800">
                        <th className="p-4 text-white">Pair</th>
                        <th className="p-4">Side</th>
                        <th className="p-4 text-center">Price</th>
                        <th className="p-4 text-center">Qty</th>
                        <th className="p-4 text-center">Status</th>
                        <th className="p-4 text-right">{activeTab === 'history' ? 'Time' : 'Action'}</th>
                    </tr>
                </thead>
                <tbody className="font-mono text-xs">
                    {displayOrders.length > 0 ? displayOrders.map(o => (
                        <tr key={o.orderId} className={`border-b border-slate-800/30 hover:bg-slate-800/20 ${(o as any).clientOrderId?.startsWith('SMART_') || (o as any).listClientOrderId?.startsWith('LIST_SMART_') ? 'bg-blue-500/5' : ''}`}>
                            <td className="p-4 font-black text-slate-300">
                                <div className="flex items-center gap-2">
                                    {o.symbol}
                                    {((o as any).clientOrderId?.startsWith('SMART_') || (o as any).listClientOrderId?.startsWith('LIST_SMART_')) && (
                                        <span className="text-[7px] bg-blue-600 text-white px-1 py-0.5 rounded font-black">SMART</span>
                                    )}
                                </div>
                            </td>
                            <td className={`p-4 font-black ${o.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>{o.side}</td>
                            <td className="p-4 text-center text-slate-400">
                                ${o.status === 'FILLED' && (Number(o.price) === 0 || !o.price)
                                    ? (Number(o.cummulativeQuoteQty) / Number(o.executedQty)).toLocaleString(undefined, { maximumFractionDigits: 2 })
                                    : Number(o.price).toLocaleString()}
                            </td>
                            <td className="p-4 text-center text-slate-500">{o.origQty}</td>
                            <td className="p-4 text-center">
                                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                    o.status === 'FILLED' ? 'bg-emerald-900/30 text-emerald-500' : 
                                    o.status === 'CANCELED' ? 'bg-rose-900/30 text-rose-500' :
                                    'bg-slate-800 text-slate-500'
                                }`}>
                                    {o.status}
                                </span>
                            </td>
                            <td className="p-4 text-right">
                                {activeTab === 'history' ? (
                                    <span className="text-slate-600 text-[10px] whitespace-nowrap">
                                        {new Date(o.updateTime).toLocaleDateString([], { day: '2-digit', month: '2-digit', year: '2-digit' })} {new Date(o.updateTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                ) : (
                                    <button onClick={() => handleCancelOrder(o.orderId, o.symbol)} className="text-rose-500 hover:text-rose-400 text-[10px] font-black uppercase tracking-widest border border-rose-500/20 px-3 py-1 rounded-lg">Cancel</button>
                                )}
                            </td>
                        </tr>
                    )) : (
                        <tr><td colSpan={6} className="py-16 text-center text-slate-700 text-xs font-black uppercase tracking-widest opacity-30">No {activeTab === 'history' ? 'Recent' : (activeTab === 'global' ? 'Global' : 'Symbol')} Orders</td></tr>
                    )}
                </tbody>
            </table>
        </div>
    );

    return (
        <div className="bg-slate-950 h-full rounded-2xl border border-slate-800 shadow-xl flex flex-col overflow-hidden">
            <div className="flex items-center px-4 border-b border-slate-800 bg-slate-900/30">
                <div className="flex gap-1 py-2">
                    <button onClick={() => setActiveTab('orders')} className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'orders' ? 'bg-blue-600 text-white' : 'text-slate-500 hover:text-white'}`}>
                        <ListFilter size={14} /> Symbol
                    </button>
                    <button onClick={() => setActiveTab('global')} className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'global' ? 'bg-indigo-600 text-white' : 'text-slate-500 hover:text-white'}`}>
                        <LayoutDashboard size={14} /> All Orders
                    </button>
                    <button onClick={() => setActiveTab('history')} className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === 'history' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-white'}`}>
                        <History size={14} /> History
                    </button>
                </div>
                <div className="flex-grow"></div>
                {activeTab === 'orders' && (
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] text-slate-600 font-black uppercase tracking-widest">Filter By Selected</span>
                        <input type="checkbox" checked={filterOrdersBySymbol} onChange={() => setFilterOrdersBySymbol(!filterOrdersBySymbol)} className="w-3 h-3 accent-blue-500" />
                    </div>
                )}
            </div>
            <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-800">
                <OrderTable />
            </div>
        </div>
    );
}


function App() {
  const [balances, setBalances] = useState([]);
  const [openOrders, setOpenOrders] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [symbols, setSymbols] = useState(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [tradingMode, setTradingMode] = useState('SPOT'); // 'SPOT' or 'LEAD'
  const [symbolSearch, setSymbolSearch] = useState('BTCUSDT');
  const [interval, setIntervalTime] = useState('1h');
  const [showChartTargets, setShowChartTargets] = useState(true);
  const [emaSettings, setEmaSettings] = useState([
    { id: 1, period: 20, color: '#3b82f6', enabled: true },
    { id: 2, period: 50, color: '#f97316', enabled: true }
  ]);
  const [globalPrices, setGlobalPrices] = useState<any[]>([]);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [price24hAgo, setPrice24hAgo] = useState(0);
  const [filterOrdersBySymbol, setFilterOrdersBySymbol] = useState(true);
  const [error, setError] = useState(null);
  const [isTrading, setIsTrading] = useState(false);
  
  const [tpPrice, setTpPrice] = useState(0);
  const [slPrice, setSlPrice] = useState(0);
  const [tpPercent, setTpPercent] = useState(2);
  const [slPercent, setSlPercent] = useState(-1);
  const [tpEnabled, setTpEnabled] = useState(true);
  const [slEnabled, setSlEnabled] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const timeframes = [
    { label: '1m', value: '1m' }, { label: '5m', value: '5m' },
    { label: '15m', value: '15m' }, { label: '1h', value: '1h' },
    { label: '4h', value: '4h' }, { label: '1d', value: '1d' },
    { label: '1w', value: '1w' }, { label: '1M', value: '1M' },
  ];

  const filteredSymbols = useMemo(() => 
    symbols.filter(s => s.toLowerCase().includes(symbolSearch.toLowerCase())),
    [symbols, symbolSearch]
  );

  const assetBalance = useMemo(() => {
    const quoteAsset = symbol.endsWith('USDT') ? 'USDT' : 'USDC';
    const asset = symbol.replace(quoteAsset, '');
    const b = balances.find(b => b.asset === asset);
    return b ? parseFloat(b.free) : 0;
  }, [balances, symbol]);
  
  const quoteBalance = useMemo(() => {
    const quoteAsset = symbol.endsWith('USDT') ? 'USDT' : 'USDC';
    const b = balances.find(b => b.asset === quoteAsset);
    return b ? parseFloat(b.free).toFixed(2) : "0.00";
  }, [balances, symbol]);

  const quoteAsset = useMemo(() => symbol.endsWith('USDT') ? 'USDT' : 'USDC', [symbol]);

  useEffect(() => {
    const fetchGlobalPrices = async () => {
      try {
        const res = await fetch('https://api.binance.com/api/v3/ticker/price');
        if (res.ok) setGlobalPrices(await res.json());
      } catch (e) {}
    };
    fetchGlobalPrices();
    const interval = setInterval(fetchGlobalPrices, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchPrivateData = async () => {
    try {
      const bRes = await fetch(`${API_BASE}/account/balances`);
      if (bRes.ok) setBalances(await bRes.json());

      const oRes = await fetch(`${API_BASE}/trades/open`);
      if (oRes.ok) setOpenOrders(await oRes.json());

      const hRes = await fetch(`${API_BASE}/trades/history?symbol=${symbol}`);
      if (hRes.ok) setTradeHistory(await hRes.json());
      
      setError(null);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    const fetchGlobalData = async () => {
        try {
          const sRes = await fetch(`${API_BASE}/symbols`);
          if (sRes.ok) setSymbols(await sRes.json());
        } catch (err) { console.error("Error fetching symbols:", err); }
    };
    fetchGlobalData();
  }, []);

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const res = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${symbol}`);
        if(res.ok) setCurrentPrice(parseFloat((await res.json()).price));

        const timestamp24hAgo = Date.now() - 24 * 60 * 60 * 1000;
        const histRes = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=1m&limit=1&startTime=${timestamp24hAgo}`);
        if(histRes.ok) {
          const histData = await histRes.json();
          if (histData && histData[0]) setPrice24hAgo(parseFloat(histData[0][4]));
        }
      } catch (err) { console.error("Price error:", err); }
    };
    fetchPrice();
    const priceInterval = setInterval(fetchPrice, 5000);
    return () => clearInterval(priceInterval);
  }, [symbol]);

  const priceChangeColor = useMemo(() => {
    if (!currentPrice || !price24hAgo) return 'text-white';
    return currentPrice >= price24hAgo ? 'text-emerald-400' : 'text-rose-400';
  }, [currentPrice, price24hAgo]);

  useEffect(() => {
    fetchPrivateData();
    const privateDataInterval = setInterval(fetchPrivateData, 10000);
    return () => clearInterval(privateDataInterval);
  }, [symbol]);

  const handleSmartTrade = async (tradeParams) => {
    if (isTrading) return;
    setIsTrading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/trades/smart-trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symbol, 
          quantity: tradeParams.quantity,
          buy_price: tradeParams.buy_price, // Optional
          take_profit_price: tradeParams.tpPrice,
          stop_loss_price: tradeParams.slPrice,
          side: tradeParams.side,
          mode: tradeParams.mode
        })
      });
      if (!res.ok) {
        const errorData = await res.json();
        console.error("Trade Failed:", errorData.detail || "Unknown error");
        setError(errorData.detail || "Trade failed");
        setIsTrading(false);
        return;
      }
      console.log("Smart Trade Created successfully!");
      fetchPrivateData();
      setIsTrading(false);
    } catch (err) { 
      setError("Trade failed: " + err.message);
      setIsTrading(false);
    }
  };

  const handleCancelOrder = async (orderId: number, orderSymbol?: string) => {
    try {
        const res = await fetch(`${API_BASE}/trades/order`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: orderSymbol || symbol, orderId })
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Cancel failed");
        alert('Order Cancelled!');
        fetchPrivateData();
    } catch (err) {
        alert("Cancel failed: " + err.message);
    }
  }

  const handleMarketClose = async () => {
    if (confirm(`Are you sure you want to market sell all ${assetBalance} ${symbol.replace('USDC', '')}?`)) {
        try {
            const res = await fetch(`${API_BASE}/trades/market-close`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol })
            });
            if (!res.ok) throw new Error((await res.json()).detail || "Market Close failed");
            alert('Position Closed!');
            fetchPrivateData();
        } catch (err) { alert("Close failed: " + err.message); }
    }
  }

  const onAutoTrade = (setup) => {
    setSymbol(setup.pair);
    setSymbolSearch(setup.pair);
    
    const entry = setup.entry || currentPrice;
    
    if (setup.tp) { 
      setTpPrice(setup.tp); 
      setTpEnabled(true);
      if (entry > 0) {
        const percent = (setup.tp - entry) / entry * 100;
        setTpPercent(parseFloat(percent.toFixed(4)));
      }
    }
    
    if (setup.sl) { 
      setSlPrice(setup.sl); 
      setSlEnabled(true);
      if (entry > 0) {
        const percent = (setup.sl - entry) / entry * 100;
        setSlPercent(parseFloat(percent.toFixed(4)));
      }
    }
  }

  const handleSelectSymbol = (s) => {
    setSymbol(s);
    setSymbolSearch(s);
  }

  return (
    <BrowserRouter>
      <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">
        <Sidebar isSidebarOpen={isSidebarOpen} setIsSidebarOpen={setIsSidebarOpen} />
        <main className="flex-1 flex flex-col min-w-0">
          <header className="flex justify-between items-center px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm z-10">
              <div className="flex-1">
                  <div className="flex items-center gap-6 relative">
                      <div className="relative group w-72">
                      <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-slate-500 group-focus-within:text-blue-400">
                        <ScanSearch size={16} />
                      </div>
                      <input 
                          type="text"
                          placeholder="Search Pair (e.g. BTC)"
                          value={symbolSearch}
                          onChange={e => setSymbolSearch(e.target.value)}
                          onFocus={() => setSymbolSearch('')}
                          className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2 text-sm outline-none focus:border-blue-500/50 font-bold transition-all focus:ring-1 focus:ring-blue-500/20"
                      />
                      <div className="absolute top-full left-0 w-full bg-slate-900 border border-slate-800 mt-2 rounded-xl shadow-2xl z-50 max-h-80 overflow-y-auto hidden group-focus-within:block border-t-0 rounded-t-none">
                          {filteredSymbols.map(s => (
                          <div 
                              key={s} 
                              onMouseDown={() => { setSymbol(s); setSymbolSearch(s); }}
                              className="px-4 py-3 hover:bg-slate-800 cursor-pointer text-sm font-bold border-b border-slate-800/50 last:border-0 transition-colors text-white"
                          >
                              {s}
                          </div>
                          ))}
                      </div>
                      </div>
                      <div className="flex flex-col">
                        <span className={`text-3xl font-mono font-black tracking-tighter transition-colors duration-500 leading-none ${priceChangeColor}`}>
                          ${currentPrice.toLocaleString(undefined, { 
                            minimumFractionDigits: currentPrice < 1 ? 6 : 2, 
                            maximumFractionDigits: currentPrice < 1 ? 8 : 2 
                          })}
                        </span>
                        {price24hAgo > 0 && (
                          <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest mt-1 opacity-60">
                            24H Change: <span className={currentPrice >= price24hAgo ? 'text-emerald-500' : 'text-rose-500'}>
                              {((currentPrice - price24hAgo) / price24hAgo * 100).toFixed(2)}%
                            </span>
                          </span>
                        )}
                      </div>
                      {error && <p className="text-rose-500 text-[10px] uppercase font-black animate-pulse ml-4 tracking-widest">⚠️ {error}</p>}
                  </div>
              </div>
              <div className="flex gap-4 items-center">
                  {/* Universal Terminal Toggle */}
                  <div className="flex bg-slate-950 p-1 rounded-2xl border border-slate-800 shadow-inner mr-4">
                    <button 
                      onClick={() => setTradingMode('SPOT')}
                      className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300 flex items-center gap-2 ${
                        tradingMode === 'SPOT' 
                          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40' 
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      <div className={`w-1.5 h-1.5 rounded-full ${tradingMode === 'SPOT' ? 'bg-blue-200 animate-pulse' : 'bg-slate-700'}`}></div>
                      Spot Mode
                    </button>
                    <button 
                      onClick={() => setTradingMode('LEAD')}
                      className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300 flex items-center gap-2 ${
                        tradingMode === 'LEAD' 
                          ? 'bg-orange-600 text-white shadow-lg shadow-orange-900/40' 
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      <div className={`w-1.5 h-1.5 rounded-full ${tradingMode === 'LEAD' ? 'bg-orange-200 animate-pulse' : 'bg-slate-700'}`}></div>
                      Lead Trading
                    </button>
                  </div>

                  <div className="flex gap-2">
                    {/* BOX 1: BASE ASSET */}
                    <div className="bg-slate-950 px-5 py-2 rounded-xl border border-slate-800 flex flex-col items-end justify-center min-w-[130px] h-[52px]">
                        <span className="text-slate-500 text-[8px] font-black uppercase tracking-[0.2em] mb-0.5 opacity-70">{symbol.replace(quoteAsset, '')} Balance</span>
                        <p className="font-mono text-sm font-black text-white tracking-tighter">{parseFloat(assetBalance.toString()).toLocaleString(undefined, { minimumFractionDigits: 4 })}</p>
                    </div>
                    {/* BOX 2: QUOTE ASSET */}
                    <div className="bg-slate-950 px-5 py-2 rounded-xl border border-slate-800 flex flex-col items-end justify-center min-w-[130px] h-[52px]">
                        <span className="text-slate-500 text-[8px] font-black uppercase tracking-[0.2em] mb-0.5 opacity-70">{quoteAsset} Balance</span>
                        <p className="font-mono text-sm font-black text-white tracking-tighter">${parseFloat(quoteBalance).toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                    </div>
                    {/* BOX 3: WALLET VALUE (MEGA TOOLTIP) */}
                    <WalletBalances balances={balances} quoteAsset={quoteAsset} globalPrices={globalPrices} />
                  </div>
              </div>
          </header>

          <div className="flex-1 min-h-0">
           <Routes>
             <Route path="/" element={
                 <div className="h-full grid grid-cols-1 xl:grid-cols-4 gap-4 p-4 xl:overflow-hidden bg-slate-900 overflow-y-auto">
                     <div className="xl:col-span-3 flex flex-col gap-4 h-full">
                         <div className="flex justify-between items-center shrink-0">
                           <div className="flex gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800/50">
                               {timeframes.map(tf => (
                               <button
                                   key={tf.value}
                                   onClick={() => setIntervalTime(tf.value)}
                                   className={`px-4 py-1 rounded-md text-[10px] font-black uppercase tracking-wider transition-all ${
                                   interval === tf.value ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'
                                   }`}
                               >
                                   {tf.label}
                               </button>
                               ))}
                               <div className="w-px h-4 bg-slate-800 mx-2 self-center"></div>
                               <button
                                   onClick={() => setShowChartTargets(!showChartTargets)}
                                   className={`px-4 py-1 rounded-md text-[10px] font-black uppercase tracking-wider transition-all ${
                                   showChartTargets ? 'text-blue-400 bg-blue-400/10' : 'text-slate-500 hover:text-slate-400'
                                   }`}
                               >
                                   {showChartTargets ? 'Targets: ON' : 'Targets: OFF'}
                               </button>
                           </div>
                           <div className="text-[10px] font-black text-slate-600 uppercase tracking-widest px-2">{symbol} • LIVE MARKET DATA</div>
                         </div>
                         <div className="flex-1 min-h-[300px] bg-slate-950 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden relative">
                             <TradingChart 
                               symbol={symbol} 
                               interval={interval} 
                               plannedTp={tpEnabled ? tpPrice : 0} 
                               plannedSl={slEnabled ? slPrice : 0} 
                               openOrders={openOrders.filter(o => o.symbol === symbol)} 
                               showTargets={showChartTargets} 
                               emaSettings={emaSettings}
                               onEmaUpdate={setEmaSettings}
                             />
                         </div>
                         <div className="h-1/3 min-h-[200px] shrink-0">
                           <BottomPanel openOrders={openOrders} tradeHistory={tradeHistory} symbol={symbol} filterOrdersBySymbol={filterOrdersBySymbol} setFilterOrdersBySymbol={setFilterOrdersBySymbol} handleCancelOrder={handleCancelOrder} />
                         </div>
                     </div>
                     <div className="xl:col-span-1 h-full min-h-[400px]">
                         <SmartTerminalView symbol={symbol} currentPrice={currentPrice} handleSmartTrade={handleSmartTrade} handleMarketClose={handleMarketClose} tpPrice={tpPrice} setTpPrice={setTpPrice} slPrice={slPrice} setSlPrice={setSlPrice} tpEnabled={tpEnabled} setTpEnabled={setTpEnabled} slEnabled={slEnabled} setSlEnabled={setSlEnabled} assetBalance={assetBalance} tradingMode={tradingMode} isTrading={isTrading} tpPercent={tpPercent} setTpPercent={setTpPercent} slPercent={slPercent} setSlPercent={setSlPercent} />
                     </div>
                 </div>
             } />
             <Route path="/scanner" element={<div className="h-full overflow-hidden bg-slate-900"><ScannerView symbols={symbols} onAutoTrade={onAutoTrade} onSelectSymbol={handleSelectSymbol} /></div>} />
             <Route path="/trades" element={<ActivePositionsView openOrders={openOrders} handleCancelOrder={handleCancelOrder} quoteBalance={quoteBalance} onSelectSymbol={handleSelectSymbol} balances={balances} globalPrices={globalPrices} />} />
             <Route path="/orders" element={<ActiveOrdersView openOrders={openOrders} handleCancelOrder={handleCancelOrder} quoteBalance={quoteBalance} onSelectSymbol={handleSelectSymbol} />} />
           </Routes>
          </div>        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
