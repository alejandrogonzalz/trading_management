import { useState, useEffect, useMemo } from 'react';
import TradingChart from './components/TradingChart';
import Sidebar from './components/Sidebar';
import { Route, Routes, HashRouter as BrowserRouter, useNavigate } from 'react-router-dom';
import ScannerView from './views/ScannerView';
import ActivePositionsView from './views/ActivePositionsView';
import { ScanSearch, ListFilter, History, LayoutDashboard } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const SmartTradePanel = ({ symbol, currentPrice, handleSmartTrade, handleMarketClose, tpPrice, setTpPrice, slPrice, setSlPrice, tpEnabled, setTpEnabled, slEnabled, setSlEnabled, assetBalance }) => {
  const [quantity, setQuantity] = useState(0.001);
  const [usdcAmount, setUsdcAmount] = useState(0);
  const [tpPercent, setTpPercent] = useState(2);
  const [slPercent, setSlPercent] = useState(-1);

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

  useEffect(() => {
    if (currentPrice > 0 && tpPrice === 0) {
      setTpPrice(Number((currentPrice * (1 + tpPercent / 100)).toFixed(2)));
    }
  }, [tpPercent, currentPrice]);

  useEffect(() => {
    if (currentPrice > 0 && slPrice === 0) {
      setSlPrice(Number((currentPrice * (1 + slPercent / 100)).toFixed(2)));
    }
  }, [slPercent, currentPrice]);

  const handleTpPriceChange = (val: number) => {
    setTpPrice(val);
    if (currentPrice > 0) {
      setTpPercent(Number(((val - currentPrice) / currentPrice * 100).toFixed(2)));
    }
  };

  const handleTpPercentChange = (val: number) => {
    setTpPercent(val);
    if (currentPrice > 0) {
      setTpPrice(Number((currentPrice * (1 + val / 100)).toFixed(2)));
    }
  };

  const handleSlPriceChange = (val: number) => {
    setSlPrice(val);
    if (currentPrice > 0) {
      setSlPercent(Number(((val - currentPrice) / currentPrice * 100).toFixed(2)));
    }
  };

  const handleSlPercentChange = (val: number) => {
    setSlPercent(val);
    if (currentPrice > 0) {
      setSlPrice(Number((currentPrice * (1 + val / 100)).toFixed(2)));
    }
  };
  
  const onTradeClick = () => {
    handleSmartTrade({
      quantity,
      tpPrice: tpEnabled ? tpPrice : 0,
      slPrice: slEnabled ? slPrice : 0,
      side: 'BUY',
    });
  }

  return (
    <div className="bg-slate-950 p-6 h-full rounded-2xl border border-slate-800 shadow-xl overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
        <h2 className="text-xl font-black mb-6 tracking-tight flex items-center gap-2 text-white uppercase">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            Smart Buy Strategy
        </h2>
      
      <div className="space-y-6">
        <div className="grid grid-cols-1 gap-4">
          <div className="space-y-2">
            <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest ml-1">Spend (USDC)</label>
            <input 
              type="number" value={usdcAmount} onChange={e => handleUsdcChange(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-lg font-mono outline-none focus:border-blue-500 transition-all text-white font-black"
            />
          </div>
          <div className="space-y-2">
            <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest ml-1">Receive ({symbol.replace('USDC', '')})</label>
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
                <div className="flex items-center gap-1">
                  <input 
                    type="number" value={tpPercent} onChange={e => handleTpPercentChange(Number(e.target.value))}
                    className="w-14 bg-slate-900/50 border border-emerald-500/20 rounded px-2 py-0.5 text-emerald-400 font-black text-xs outline-none text-right"
                  />
                  <span className="text-[10px] text-emerald-500 font-black">%</span>
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
                <div className="flex items-center gap-1">
                  <input 
                    type="number" value={slPercent} onChange={e => handleSlPercentChange(Number(e.target.value))}
                    className="w-14 bg-slate-900/50 border border-rose-500/20 rounded px-2 py-0.5 text-rose-400 font-black text-xs outline-none text-right"
                  />
                  <span className="text-[10px] text-rose-500 font-black">%</span>
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
          className="w-full bg-blue-600 hover:bg-blue-500 text-white font-black py-4 rounded-2xl mt-4 transition-all shadow-lg shadow-blue-900/30 uppercase tracking-widest text-sm"
        >
          Execute Smart Trade
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
        if (activeTab === 'global') return openOrders;
        return filterOrdersBySymbol ? openOrders.filter(o => o.symbol === symbol) : openOrders;
    }, [openOrders, symbol, filterOrdersBySymbol, activeTab]);

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
                        <th className="p-4 text-right">Action</th>
                    </tr>
                </thead>
                <tbody className="font-mono text-xs">
                    {displayOrders.length > 0 ? displayOrders.map(o => (
                        <tr key={o.orderId} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                            <td className="p-4 font-black text-slate-300">{o.symbol}</td>
                            <td className={`p-4 font-black ${o.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>{o.side}</td>
                            <td className="p-4 text-center text-slate-400">${Number(o.price).toLocaleString()}</td>
                            <td className="p-4 text-center text-slate-500">{o.origQty}</td>
                            <td className="p-4 text-center"><span className="px-2 py-0.5 rounded-full bg-slate-800 text-[10px] font-bold text-slate-500">{o.status}</span></td>
                            <td className="p-4 text-right">
                                <button onClick={() => handleCancelOrder(o.orderId, o.symbol)} className="text-rose-500 hover:text-rose-400 text-[10px] font-black uppercase tracking-widest border border-rose-500/20 px-3 py-1 rounded-lg">Cancel</button>
                            </td>
                        </tr>
                    )) : (
                        <tr><td colSpan={6} className="py-16 text-center text-slate-700 text-xs font-black uppercase tracking-widest opacity-30">No Active {activeTab === 'global' ? 'USDC' : ''} Orders</td></tr>
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
                        <LayoutDashboard size={14} /> All USDC
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
  const [symbols, setSymbols] = useState(['BTCUSDC', 'ETHUSDC', 'SOLUSDC']);
  const [symbol, setSymbol] = useState('BTCUSDC');
  const [symbolSearch, setSymbolSearch] = useState('BTCUSDC');
  const [interval, setIntervalTime] = useState('1h');
  const [currentPrice, setCurrentPrice] = useState(0);
  const [price24hAgo, setPrice24hAgo] = useState(0);
  const [filterOrdersBySymbol, setFilterOrdersBySymbol] = useState(true);
  const [error, setError] = useState(null);
  
  const [tpPrice, setTpPrice] = useState(0);
  const [slPrice, setSlPrice] = useState(0);
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
    const asset = symbol.replace('USDC', '');
    const b = balances.find(b => b.asset === asset);
    return b ? parseFloat(b.free) : 0;
  }, [balances, symbol]);
  
  const usdcBalance = useMemo(() => {
    const b = balances.find(b => b.asset === 'USDC');
    return b ? parseFloat(b.free).toFixed(2) : "0.00";
  }, [balances]);

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
    try {
      const res = await fetch(`${API_BASE}/trades/smart-trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, side: 'BUY', ...tradeParams })
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Trade failed");
      alert(`Smart Trade Created!`);
      fetchPrivateData();
    } catch (err) { alert("Trade failed: " + err.message); }
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
    if (setup.tp) { setTpPrice(setup.tp); setTpEnabled(true); }
    if (setup.sl) { setSlPrice(setup.sl); setSlEnabled(true); }
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
                          ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
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
              <div className="flex gap-4">
                  <div className="bg-blue-600/5 px-5 py-2 rounded-xl border border-blue-500/20 flex flex-col items-end justify-center">
                      <span className="text-blue-400 text-[9px] font-black uppercase tracking-[0.2em] mb-0.5 opacity-70">USDC Balance</span>
                      <p className="font-mono text-xl font-black text-blue-100 tracking-tighter">${parseFloat(usdcBalance).toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                  </div>
              </div>
          </header>

          <div className="flex-1 min-h-0">
            <Routes>
              <Route path="/" element={
                  <div className="h-full grid grid-cols-1 xl:grid-cols-4 gap-4 p-4 overflow-y-auto xl:overflow-hidden bg-slate-900">
                      <div className="xl:col-span-3 flex flex-col gap-4 min-h-[600px]">
                          <div className="flex justify-between items-center">
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
                            </div>
                            <div className="text-[10px] font-black text-slate-600 uppercase tracking-widest px-2">{symbol} • LIVE MARKET DATA</div>
                          </div>
                          <div className="flex-1 min-h-[400px] bg-slate-950 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden relative">
                              <TradingChart symbol={symbol} interval={interval} plannedTp={tpEnabled ? tpPrice : 0} plannedSl={slEnabled ? slPrice : 0} openOrders={openOrders.filter(o => o.symbol === symbol)} />
                          </div>
                          <div className="xl:h-1/3 min-h-[250px]">
                            <BottomPanel openOrders={openOrders} tradeHistory={tradeHistory} symbol={symbol} filterOrdersBySymbol={filterOrdersBySymbol} setFilterOrdersBySymbol={setFilterOrdersBySymbol} handleCancelOrder={handleCancelOrder} />
                          </div>
                      </div>
                      <div className="xl:col-span-1 min-h-[500px]">
                          <SmartTradePanel symbol={symbol} currentPrice={currentPrice} handleSmartTrade={handleSmartTrade} handleMarketClose={handleMarketClose} tpPrice={tpPrice} setTpPrice={setTpPrice} slPrice={slPrice} setSlPrice={setSlPrice} tpEnabled={tpEnabled} setTpEnabled={setTpEnabled} slEnabled={slEnabled} setSlEnabled={setSlEnabled} assetBalance={assetBalance} />
                      </div>
                  </div>
              } />
              <Route path="/scanner" element={<div className="h-full overflow-hidden bg-slate-900"><ScannerView symbols={symbols} onAutoTrade={onAutoTrade} /></div>} />
              <Route path="/positions" element={<ActivePositionsView openOrders={openOrders} handleCancelOrder={handleCancelOrder} usdcBalance={usdcBalance} />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
