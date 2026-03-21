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
  
  // SHARED STATE
  const [symbols, setSymbols] = useState(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);
  const [leadWhitelist, setLeadWhitelist] = useState<string[]>([]);
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [tradingMode, setTradingMode] = useState('SPOT'); // 'SPOT' or 'LEAD'
  const [symbolSearch, setSymbolSearch] = useState('BTCUSDT');
  const [globalPrices, setGlobalPrices] = useState<any[]>([]);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [price24hAgo, setPrice24hAgo] = useState(0);
  const [isTrading, setIsTrading] = useState(false);
  const [error, setError] = useState(null);

  // ISOLATED DATA STATE
  const [spotBalances, setSpotBalances] = useState([]);
  const [leadBalances, setLeadBalances] = useState([]);
  const [spotOrders, setSpotOrders] = useState([]);
  const [leadOrders, setLeadOrders] = useState([]);
  const [spotHistory, setSpotHistory] = useState([]);
  const [leadHistory, setLeadHistory] = useState([]);
  const [leadPositions, setLeadPositions]    = useState([]);

  // UI STATE
  const [interval, setIntervalTime] = useState('1h');
  const [showChartTargets, setShowChartTargets] = useState(true);
  const [emaSettings, setEmaSettings] = useState([
    { id: 1, period: 20, color: '#3b82f6', enabled: true },
    { id: 2, period: 50, color: '#f97316', enabled: true }
  ]);
  const [filterOrdersBySymbol, setFilterOrdersBySymbol] = useState(true);
  const [tpPrice, setTpPrice] = useState(0);
  const [slPrice, setSlPrice] = useState(0);
  const [tpPercent, setTpPercent] = useState(2);
  const [slPercent, setSlPercent] = useState(1);
  const [tpEnabled, setTpEnabled] = useState(true);
  const [slEnabled, setSlEnabled] = useState(true);
  const [tradeSide, setTradeSide] = useState('BUY');
  const [tradeLeverage, setTradeLeverage] = useState(10);
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // ... (rest of the states)

  const timeframes = [
    { label: '1m', value: '1m' }, { label: '5m', value: '5m' },
    { label: '15m', value: '15m' }, { label: '1h', value: '1h' },
    { label: '4h', value: '4h' }, { label: '1d', value: '1d' },
    { label: '1w', value: '1w' }, { label: '1M', value: '1M' },
  ];

  const filteredSymbols = useMemo(() => {
    const list = (tradingMode === 'LEAD' && leadWhitelist.length > 0) ? leadWhitelist : symbols;
    return list.filter(s => s.toLowerCase().includes(symbolSearch.toLowerCase()));
  }, [symbols, leadWhitelist, symbolSearch, tradingMode]);

  // DERIVED DATA FOR COMPONENTS
  const activeBalances = useMemo(() => tradingMode === 'LEAD' ? leadBalances : spotBalances, [tradingMode, leadBalances, spotBalances]);
  const activeOrders = useMemo(() => tradingMode === 'LEAD' ? leadOrders : spotOrders, [tradingMode, leadOrders, spotOrders]);
  const activeHistory = useMemo(() => tradingMode === 'LEAD' ? leadHistory : spotHistory, [tradingMode, leadHistory, spotHistory]);

  const quoteAsset = useMemo(() => symbol.endsWith('USDT') ? 'USDT' : 'USDC', [symbol]);

  const assetBalance = useMemo(() => {
    const asset = symbol.replace(quoteAsset, '');
    const b = activeBalances.find(b => b.asset === asset);
    if (!b) return 0;
    return tradingMode === 'LEAD' ? parseFloat(b.availableBalance || 0) : parseFloat(b.free || 0);
  }, [activeBalances, symbol, quoteAsset, tradingMode]);

  const quoteBalance = useMemo(() => {
    const b = activeBalances.find(b => b.asset === quoteAsset);
    if (!b) return "0.00";
    const val = tradingMode === 'LEAD' ? parseFloat(b.availableBalance || 0) : parseFloat(b.free || 0);
    return val.toFixed(2);
  }, [activeBalances, symbol, quoteAsset, tradingMode]);

  const fetchPrivateData = async () => {
    const isLead = tradingMode === 'LEAD';
    try {
      if (isLead) {
        const [bRes, oRes, pRes, hRes] = await Promise.all([
          fetch(`${API_BASE}/lead/balances`),
          fetch(`${API_BASE}/lead/open-orders`),
          fetch(`${API_BASE}/lead/positions`),
          fetch(`${API_BASE}/lead/history`)
        ]);
        if (bRes.ok) { const data = await bRes.json(); setLeadBalances(data.assets || []); }
        if (oRes.ok) setLeadOrders(await oRes.json());
        if (pRes.ok) setLeadPositions(await pRes.json());
        if (hRes.ok) setLeadHistory(await hRes.json());
      } else {
        const [bRes, oRes, hRes] = await Promise.all([
          fetch(`${API_BASE}/account/balances`),
          fetch(`${API_BASE}/trades/open`),
          fetch(`${API_BASE}/trades/smart-history`)
        ]);
        if (bRes.ok) setSpotBalances(await bRes.json());
        if (oRes.ok) setSpotOrders(await oRes.json());
        if (hRes.ok) setSpotHistory(await hRes.json());
      }
    } catch (err) { console.error("Private data fetch failed:", err); }
  };

  useEffect(() => {
    fetchPrivateData();
    const interval = setInterval(fetchPrivateData, 5000);
    return () => clearInterval(interval);
  }, [tradingMode, symbol]);

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

  useEffect(() => {
    const fetchGlobalData = async () => {
        try {
          const sRes = await fetch(`${API_BASE}/symbols`);
          if (sRes.ok) setSymbols(await sRes.json());
          
          const lRes = await fetch(`${API_BASE}/lead/symbols`);
          if (lRes.ok) {
            const raw = await lRes.json();
            const list = Array.isArray(raw.data) 
                ? raw.data.map((item: any) => typeof item === 'string' ? item : item.symbol)
                : (Array.isArray(raw) ? raw : []);
            setLeadWhitelist(list);
          }
        } catch (err) { console.warn("Symbols fetch failed", err); }
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

  const handleSmartTrade = async (tradeData) => {
    setIsTrading(true);
    try {
      const isLead = tradingMode === 'LEAD';
      const endpoint = isLead ? 'lead/smart-order' : 'trades/smart-trade';
      
      const payload: any = {
        symbol,
        quantity: tradeData.quantity,
        take_profit_price: tradeData.tpPrice,
        stop_loss_price: tradeData.slPrice,
        side: tradeData.side,
        mode: tradeData.mode
      };

      if (isLead) {
          payload.leverage = tradeData.leverage || 10;
          payload.type = 'MARKET';
      }

      const response = await fetch(`${API_BASE}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        await fetchPrivateData();
        setNotification({ message: `Successfully executed ${tradeData.side} ${isLead ? 'LEAD' : ''} order for ${symbol}`, type: 'success' });
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
      const isLead = tradingMode === 'LEAD';
      const endpoint = isLead ? 'lead/order' : 'trades/cancel';
      const method = isLead ? 'DELETE' : 'POST';
      
      const response = await fetch(`${API_BASE}/${endpoint}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ orderId, symbol: orderSymbol }),
      });
      if (response.ok) {
          fetchPrivateData();
          setNotification({ message: `Cancelled order for ${orderSymbol}`, type: 'success' });
      } else {
          setNotification({ message: 'Cancel failed', type: 'error' });
      }
    } catch (err) { console.error("Cancel failed", err); }
  };

  const handleMarketClose = async () => {
    const isLead = tradingMode === 'LEAD';
    const confirmMsg = isLead 
        ? `Close active LEAD position for ${symbol}?`
        : `Emergency SELL all ${assetBalance} ${symbol.replace(quoteAsset, '')}?`;

    if (window.confirm(confirmMsg)) {
        try {
            const endpoint = isLead ? 'lead/close-position' : 'trades/market-close';
            const res = await fetch(`${API_BASE}/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, quantity: isLead ? null : assetBalance })
            });
            if (res.ok) {
                fetchPrivateData();
                setNotification({ message: `Successfully closed ${isLead ? 'LEAD position' : 'SPOT holdings'} for ${symbol}`, type: 'success' });
            } else {
                const err = await res.json();
                setNotification({ message: err.detail || 'Close failed', type: 'error' });
            }
        } catch (e) { 
            console.error(e);
            setNotification({ message: 'Failed to connect to exchange', type: 'error' });
        }
    }
  };

  const onAutoTrade = (setup) => {
    setSymbol(setup.pair);
    setSymbolSearch(setup.pair);
    
    // Determine side from AI bias
    const bias = setup.bias?.toLowerCase() || 'bullish';
    const side = bias === 'bearish' ? 'SELL' : 'BUY';
    setTradeSide(side);

    // Apply recommended leverage if available
    if (setup.leverage) {
      setTradeLeverage(parseInt(setup.leverage));
    }

    // If Bearish, automatically switch to LEAD mode as SPOT doesn't support shorts
    if (side === 'SELL' && tradingMode === 'SPOT') {
      setTradingMode('LEAD');
    }

    const entry = setup.entry || currentPrice;
    if (setup.tp) { 
      setTpPrice(setup.tp); setTpEnabled(true);
      if (entry > 0) {
        const percent = Math.abs((setup.tp - entry) / entry * 100);
        setTpPercent(parseFloat(percent.toFixed(4)));
      }
    }
    if (setup.sl) { 
      setSlPrice(setup.sl); setSlEnabled(true);
      if (entry > 0) {
        const percent = Math.abs((setup.sl - entry) / entry * 100);
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
      <div className="flex h-screen bg-slate-900 text-slate-300 font-sans selection:bg-blue-500/30">
        {notification && <NotificationToast message={notification.message} type={notification.type} onClose={() => setNotification(null)} />}
        
        <Sidebar isSidebarOpen={isSidebarOpen} setIsSidebarOpen={setIsSidebarOpen} />

        <main className={`flex-1 flex flex-col min-w-0 transition-all duration-300 overflow-hidden`}>
          <header className="h-16 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-40">
              <div className="flex items-center gap-8">
                  {/* SYMBOL SEARCH */}
                  <div className="relative group w-72">
                    <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-slate-500 group-focus-within:text-blue-400 transition-colors">
                      <ScanSearch size={16} />
                    </div>
                    <input 
                      type="text" 
                      placeholder="Search pair..."
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2 pl-10 pr-4 text-xs font-black text-white focus:outline-none focus:border-blue-500/50 focus:ring-4 focus:ring-blue-500/5 transition-all uppercase tracking-widest"
                      value={symbolSearch}
                      onChange={(e) => setSymbolSearch(e.target.value.toUpperCase())}
                      onFocus={() => setIsSearchFocused(true)}
                      onBlur={() => setTimeout(() => setIsSearchFocused(false), 150)}
                    />
                    {isSearchFocused && filteredSymbols.length > 0 && (
                      <div className="absolute top-full left-0 right-0 mt-2 bg-slate-950 border border-slate-800 rounded-xl shadow-2xl overflow-hidden z-50 max-h-64 overflow-y-auto custom-scrollbar backdrop-blur-xl">
                        {filteredSymbols.map(s => (
                          <button 
                            key={s}
                            onClick={() => handleSelectSymbol(s)}
                            className="w-full px-4 py-3 text-left text-xs font-black text-slate-300 hover:bg-blue-600 hover:text-white border-b border-slate-800/50 last:border-0 transition-colors flex justify-between items-center"
                          >
                            {s}
                            <span className="text-[8px] opacity-50 font-black">BINANCE</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-4">
                      <div className="flex flex-col">
                          <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest leading-none mb-1">Live Price</span>
                          <div className={`text-lg font-black font-mono tracking-tighter ${priceChangeColor} leading-none`}>
                              {currentPrice > 0 ? `$${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 8 })}` : 'SYNCING...'}
                          </div>
                      </div>
                  </div>
              </div>

              <div className="flex items-center gap-6">
                  {/* MODE SWITCHER */}
                  <div className="bg-slate-950 p-1 rounded-2xl border border-slate-800 flex shadow-inner">
                      <button 
                        onClick={() => setTradingMode('SPOT')}
                        className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300 flex items-center gap-2 ${tradingMode === 'SPOT' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40' : 'text-slate-500 hover:text-slate-300'}`}
                      >
                        <div className={`w-1.5 h-1.5 rounded-full ${tradingMode === 'SPOT' ? 'bg-blue-200 animate-pulse' : 'bg-slate-700'}`}></div>
                        Spot
                      </button>
                      <button 
                        onClick={() => setTradingMode('LEAD')}
                        className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300 flex items-center gap-2 ${tradingMode === 'LEAD' ? 'bg-orange-600 text-white shadow-lg shadow-orange-900/40' : 'text-slate-500 hover:text-slate-300'}`}
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
                    <WalletBalances balances={activeBalances} quoteAsset={quoteAsset} globalPrices={globalPrices} mode={tradingMode} />
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
                               openOrders={activeOrders.filter(o => o.symbol === symbol)} 
                               showTargets={showChartTargets} 
                               emaSettings={emaSettings}
                               onEmaUpdate={setEmaSettings}
                             />
                         </div>
                         <div className="flex-1 min-h-[250px] shrink-0">
                           <BottomPanel openOrders={activeOrders} tradeHistory={activeHistory} symbol={symbol} filterOrdersBySymbol={filterOrdersBySymbol} setFilterOrdersBySymbol={setFilterOrdersBySymbol} handleCancelOrder={handleCancelOrder} tradingMode={tradingMode} />
                         </div>
                     </div>
                     <div className="xl:col-span-1 h-full min-h-[400px]">
                         <SmartTerminalView symbol={symbol} currentPrice={currentPrice} handleSmartTrade={handleSmartTrade} handleMarketClose={handleMarketClose} tpPrice={tpPrice} setTpPrice={setTpPrice} slPrice={slPrice} setSlPrice={setSlPrice} tpEnabled={tpEnabled} setTpEnabled={setTpEnabled} slEnabled={slEnabled} setSlEnabled={setSlEnabled} assetBalance={assetBalance} tradingMode={tradingMode} isTrading={isTrading} tpPercent={tpPercent} setTpPercent={setTpPercent} slPercent={slPercent} setSlPercent={setSlPercent} side={tradeSide} setSide={setTradeSide} leverage={tradeLeverage} setLeverage={setTradeLeverage} />
                     </div>
                 </div>
             } />
             <Route path="/scanner" element={<div className="h-full overflow-hidden bg-slate-900"><ScannerView symbols={symbols} onAutoTrade={onAutoTrade} onSelectSymbol={handleSelectSymbol} /></div>} />
             <Route path="/trades" element={<ActivePositionsView openOrders={activeOrders} handleCancelOrder={handleCancelOrder} quoteBalance={quoteBalance} onSelectSymbol={handleSelectSymbol} balances={activeBalances} globalPrices={globalPrices} tradingMode={tradingMode} leadPositions={leadPositions} smartHistory={activeHistory} />} />
             <Route path="/orders" element={<ActiveOrdersView openOrders={activeOrders} handleCancelOrder={handleCancelOrder} quoteBalance={quoteBalance} onSelectSymbol={handleSelectSymbol} tradingMode={tradingMode} />} />
           </Routes>
          </div>        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
