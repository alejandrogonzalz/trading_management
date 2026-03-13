import { useState, useEffect, useMemo } from 'react';
import TradingChart from './components/TradingChart';
import Sidebar from './components/Sidebar';
import { Route, Routes, HashRouter as BrowserRouter } from 'react-router-dom';
import ScannerView from './views/ScannerView';
import ActivePositionsView from './views/ActivePositionsView';
import ActiveOrdersView from './views/ActiveOrdersView';
import SmartTerminalView from './views/SmartTerminalView';
import BottomPanel from './components/BottomPanel';
import WalletBalances from './components/WalletBalances';
import { ScanSearch, Zap, CheckCircle2, AlertCircle } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const NotificationToast = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === 'success' ? 'bg-emerald-500/10 border-emerald-500/50' : 'bg-rose-500/10 border-rose-500/50';
  const iconColor = type === 'success' ? 'text-emerald-400' : 'text-rose-400';

  return (
    <div className={`fixed bottom-8 right-8 z-[5000] px-6 py-4 rounded-2xl border backdrop-blur-xl shadow-2xl flex items-center gap-4 animate-in slide-in-from-right-8 duration-500 ${bgColor}`}>
      {type === 'success' ? <CheckCircle2 className={iconColor} size={24} /> : <AlertCircle className={iconColor} size={24} />}
      <div>
        <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest leading-none mb-1">{type === 'success' ? 'Transaction Complete' : 'System Error'}</p>
        <p className="text-sm font-black text-white">{message}</p>
      </div>
      <button onClick={onClose} className="ml-4 text-slate-500 hover:text-white transition-colors outline-none">
        <Zap size={14} className="rotate-90" />
      </button>
    </div>
  );
};

function App() {
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'error'} | null>(null);
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
        if (res.ok) {
          const data = await res.json();
          setCurrentPrice(parseFloat(data.price));
        }
        const res2 = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=1d&limit=2`);
        if (res2.ok) {
          const data2 = await res2.json();
          if (data2.length > 0) setPrice24hAgo(parseFloat(data2[0][4]));
        }
      } catch (err) { console.error("Price fetch failed:", err); }
    };
    fetchPrice();
    const intervalId = setInterval(fetchPrice, 3000);
    return () => clearInterval(intervalId);
  }, [symbol]);

  useEffect(() => {
    fetchPrivateData();
    const intervalId = setInterval(fetchPrivateData, 5000);
    return () => clearInterval(intervalId);
  }, [symbol]);

  const handleSmartTrade = async (tradeData) => {
    setIsTrading(true);
    try {
      const response = await fetch(`${API_BASE}/trades/smart-trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          quantity: tradeData.quantity,
          take_profit_price: tradeData.tpPrice,
          stop_loss_price: tradeData.slPrice,
          side: tradeData.side,
          mode: tradeData.mode
        }),
      });
      if (response.ok) {
        await fetchPrivateData();
        setNotification({ message: `Successfully executed ${tradeData.side} order for ${symbol}`, type: 'success' });
      } else {
        const err = await response.json();
        setNotification({ message: err.detail || 'Trade execution failed', type: 'error' });
      }
    } catch (err) { 
      setNotification({ message: 'Failed to connect to exchange gateway', type: 'error' });
    }
    finally { setIsTrading(false); }
  };

  const handleCancelOrder = async (orderId, orderSymbol) => {
    try {
      const response = await fetch(`${API_BASE}/trades/order`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: orderSymbol, orderId }),
      });
      if (response.ok) fetchPrivateData();
    } catch (err) { console.error("Cancel failed:", err); }
  };

  const handleMarketClose = async () => {
    if (window.confirm(`Emergency SELL all ${assetBalance} ${symbol.replace(quoteAsset, '')}?`)) {
        try {
            const res = await fetch(`${API_BASE}/trades/market-close`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, quantity: assetBalance })
            });
            if (res.ok) fetchPrivateData();
        } catch (e) { console.error(e); }
    }
  };

  const onAutoTrade = (setup) => {
    setSymbol(setup.pair);
    setSymbolSearch(setup.pair);
    const entry = setup.entry || currentPrice;
    if (setup.tp) { 
      setTpPrice(setup.tp); setTpEnabled(true);
      if (entry > 0) {
        const percent = (setup.tp - entry) / entry * 100;
        setTpPercent(parseFloat(percent.toFixed(4)));
      }
    }
    if (setup.sl) { 
      setSlPrice(setup.sl); setSlEnabled(true);
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

  const priceChangeColor = currentPrice >= price24hAgo ? 'text-emerald-400' : 'text-rose-400';

  return (
    <BrowserRouter>
      <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">
        {notification && (
          <NotificationToast 
            message={notification.message} 
            type={notification.type} 
            onClose={() => setNotification(null)} 
          />
        )}
        <Sidebar isSidebarOpen={isSidebarOpen} setIsSidebarOpen={setIsSidebarOpen} />
        <main className="flex-1 flex flex-col min-w-0">
          <header className="flex justify-between items-center px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm z-50">
              <div className="flex items-center gap-8">
                  <div className="relative group w-72">
                      <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-slate-500 group-focus-within:text-blue-400">
                        <ScanSearch size={16} />
                      </div>
                      <input
                          type="text"
                          placeholder="Search Pair (e.g. BTC)"
                          value={symbolSearch}
                          onChange={(e) => setSymbolSearch(e.target.value.toUpperCase())}
                          className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2.5 pl-10 pr-4 text-xs font-black tracking-widest outline-none focus:border-blue-500 transition-all focus:ring-1 focus:ring-blue-500/20"
                      />
                      {symbolSearch && symbolSearch !== symbol && (
                          <div className="absolute top-full left-0 right-0 mt-2 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl z-[3000] overflow-hidden overflow-y-auto max-h-80 custom-scrollbar backdrop-blur-xl">
                              {filteredSymbols.length > 0 ? filteredSymbols.map(s => (
                                  <div
                                      key={s}
                                      onMouseDown={() => { setSymbol(s); setSymbolSearch(s); }}
                                      className="px-4 py-3 hover:bg-slate-800 cursor-pointer text-sm font-bold border-b border-slate-800/50 last:border-0 transition-colors text-white"
                                  >
                                      {s}
                                  </div>
                              )) : (
                                  <div className="px-4 py-3 text-slate-500 text-xs italic text-center">No symbols found</div>
                              )}
                          </div>
                      )}
                  </div>

                  <div className="flex items-center gap-4">
                      <div className="flex flex-col">
                          <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest leading-none mb-1">Live Price</span>
                          <div className={`text-xl font-black font-mono tracking-tighter ${priceChangeColor} leading-none`}>
                              {currentPrice > 0 ? `$${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 8 })}` : 'SYNCING...'}
                          </div>
                      </div>
                  </div>
              </div>

              <div className="flex items-center gap-6">
                  <div className="flex bg-slate-950 p-1 rounded-2xl border border-slate-800">
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
                 <div className="h-full grid grid-cols-1 xl:grid-cols-4 gap-4 p-4 xl:overflow-hidden bg-slate-900">
                     <div className="xl:col-span-3 flex flex-col gap-4 h-full overflow-hidden">
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
                         <div className="flex-[2] min-h-[350px] bg-slate-950 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden relative">
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
                         <div className="flex-1 min-h-[250px] shrink-0">
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
