import { useState, useEffect, useMemo } from 'react';
import { ArrowUp, ArrowDown, ShieldAlert, Zap, TrendingUp, TrendingDown } from 'lucide-react';

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
  const [leverage, setLeverage] = useState(10);

  // Force 'BUY' if in SPOT mode
  useEffect(() => {
    if (tradingMode === 'SPOT') {
        setSide('BUY');
        setLeverage(1);
    } else {
        setLeverage(10);
    }
  }, [tradingMode]);

  // Unified Cost Calculation (Leverage-Aware)
  const calculateQuantity = (spend: number, lev: number, price: number) => {
    if (price <= 0) return 0;
    // Total Value = Spend * Leverage
    // Quantity = Total Value / Price
    return (spend * lev) / price;
  };

  const calculateSpend = (qty: number, lev: number, price: number) => {
    if (lev <= 0 || price <= 0) return 0;
    // Total Value = qty * price
    // Spend (Margin) = Total Value / Leverage
    return (qty * price) / lev;
  };

  // Sync USDC when quantity changes
  const handleQuantityChange = (val: number) => {
    setQuantity(val);
    if (currentPrice > 0) {
      setUsdcAmount(Number(calculateSpend(val, leverage, currentPrice).toFixed(2)));
    }
  };

  // Sync Quantity when USDC changes
  const handleUsdcChange = (val: number) => {
    setUsdcAmount(val);
    if (currentPrice > 0) {
      setQuantity(Number(calculateQuantity(val, leverage, currentPrice).toFixed(6)));
    }
  };

  // Re-sync on currentPrice or leverage change
  useEffect(() => {
    if (currentPrice > 0) {
        if (usdcAmount > 0) {
            setQuantity(Number(calculateQuantity(usdcAmount, leverage, currentPrice).toFixed(6)));
        } else {
            setUsdcAmount(Number(calculateSpend(quantity, leverage, currentPrice).toFixed(2)));
        }
    }
  }, [currentPrice, leverage]);

  // Price/Percent Sync Logic
  const handleTpPercentChange = (val: number) => {
    const fixedVal = Number(val.toFixed(4));
    setTpPercent(fixedVal);
    if (currentPrice > 0) {
      const multiplier = side === 'BUY' ? (1 + fixedVal / 100) : (1 - fixedVal / 100);
      setTpPrice(Number((currentPrice * multiplier).toFixed(8)));
    }
  };

  const handleSlPercentChange = (val: number) => {
    const fixedVal = Number(val.toFixed(4));
    setSlPercent(fixedVal);
    if (currentPrice > 0) {
      const multiplier = side === 'BUY' ? (1 + fixedVal / 100) : (1 - fixedVal / 100);
      setSlPrice(Number((currentPrice * multiplier).toFixed(8)));
    }
  };

  const handleTpPriceChange = (val: number) => {
    setTpPrice(val);
    if (currentPrice > 0) {
      const diff = side === 'BUY' ? (val - currentPrice) : (currentPrice - val);
      setTpPercent(Number((diff / currentPrice * 100).toFixed(4)));
    }
  };

  const handleSlPriceChange = (val: number) => {
    setSlPrice(val);
    if (currentPrice > 0) {
      const diff = side === 'BUY' ? (val - currentPrice) : (currentPrice - val);
      setSlPercent(Number((diff / currentPrice * 100).toFixed(4)));
    }
  };
  
  const onTradeClick = () => {
    handleSmartTrade({
      quantity,
      tpPrice: tpEnabled ? tpPrice : 0,
      slPrice: slEnabled ? slPrice : 0,
      side,
      mode: tradingMode,
      leverage: tradingMode === 'LEAD' ? leverage : 1
    });
  }

  // Estimated Liquidation Price (Simple approximation for isolated margin)
  const liqPrice = useMemo(() => {
    if (tradingMode !== 'LEAD' || currentPrice <= 0 || leverage <= 1) return 0;
    const maintenanceMargin = 0.005; // 0.5% approximation
    if (side === 'BUY') {
        return currentPrice * (1 - (1 / leverage) + maintenanceMargin);
    } else {
        return currentPrice * (1 + (1 / leverage) - maintenanceMargin);
    }
  }, [currentPrice, leverage, side, tradingMode]);

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
              className={`flex-1 py-2.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${side === 'BUY' ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/40' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <TrendingUp size={14} /> Long
            </button>
            <button 
              onClick={() => setSide('SELL')}
              className={`flex-1 py-2.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${side === 'SELL' ? 'bg-rose-600 text-white shadow-lg shadow-rose-900/40' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <TrendingDown size={14} /> Short
            </button>
          </div>
        )}

        {/* Leverage Slider (Only for Lead Trading) */}
        {tradingMode === 'LEAD' && (
            <div className="space-y-3 bg-slate-900/50 p-4 rounded-2xl border border-slate-800">
                <div className="flex justify-between items-center">
                    <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Initial Leverage</label>
                    <span className="text-sm font-mono font-black text-orange-400">{leverage}x</span>
                </div>
                <input 
                    type="range" min="1" max="50" step="1" value={leverage}
                    onChange={(e) => setLeverage(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-500"
                />
                <div className="flex justify-between text-[8px] font-black text-slate-600 uppercase tracking-tighter">
                    <span>1x</span>
                    <span>10x</span>
                    <span>25x</span>
                    <span>50x</span>
                </div>
            </div>
        )}

        <div className="grid grid-cols-1 gap-4">
          <div className="space-y-2">
            <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest ml-1">Margin ({quoteAsset})</label>
            <input 
              type="number" value={usdcAmount} onChange={e => handleUsdcChange(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-lg font-mono outline-none focus:border-blue-500 transition-all text-white font-black"
            />
          </div>
          <div className="space-y-2 relative group">
            <label className="text-[10px] text-slate-500 font-black uppercase tracking-widest ml-1">Position Size ({symbol.replace(quoteAsset, '')})</label>
            <input 
              type="number" step="0.0001" value={quantity} onChange={e => handleQuantityChange(Number(e.target.value))}
              className="w-full bg-slate-800/30 border border-slate-800 rounded-xl px-4 py-3 text-lg font-mono outline-none focus:border-blue-500/50 transition-all text-slate-300"
            />
            {leverage > 1 && (
                <div className="absolute right-4 bottom-3 text-[9px] font-black text-slate-600 uppercase">
                    Value: ${(quantity * currentPrice).toFixed(2)}
                </div>
            )}
          </div>
        </div>

        {/* Risk Indicators (Futures Only) */}
        {tradingMode === 'LEAD' && liqPrice > 0 && (
            <div className="flex gap-2">
                <div className="flex-1 bg-rose-500/5 border border-rose-500/20 p-3 rounded-xl flex flex-col items-center">
                    <span className="text-[8px] font-black text-rose-500 uppercase mb-1">Est. Liquidation</span>
                    <span className="text-xs font-mono font-black text-white">${liqPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                </div>
                <div className="flex-1 bg-slate-900 border border-slate-800 p-3 rounded-xl flex flex-col items-center">
                    <span className="text-[8px] font-black text-slate-500 uppercase mb-1">Risk Buffer</span>
                    <span className="text-xs font-mono font-black text-emerald-400">
                        {Math.abs(((liqPrice - currentPrice) / currentPrice * 100)).toFixed(1)}%
                    </span>
                </div>
            </div>
        )}

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

export default SmartTerminalView;
