import { useState, useEffect, useMemo } from 'react';
import { ArrowUp, ArrowDown, ShieldAlert, Zap, TrendingUp, TrendingDown } from 'lucide-react';

const SmartTerminalView = ({ 
  symbol, currentPrice, handleSmartTrade, handleMarketClose, 
  tpPrice, setTpPrice, slPrice, setSlPrice, 
  tpEnabled, setTpEnabled, slEnabled, setSlEnabled, 
  assetBalance, tradingMode, isTrading,
  tpPercent, setTpPercent, slPercent, setSlPercent,
  side, setSide, leverage, setLeverage
}) => {
  const [quantity, setQuantity] = useState(0.001);
  const [usdcAmount, setUsdcAmount] = useState(0);

  // Force 'BUY' if in SPOT mode
  useEffect(() => {
    if (tradingMode === 'SPOT') {
        setSide('BUY');
        setLeverage(1);
    }
    // Don't force reset to 10 for LEAD, respect current or AI-set value
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
  const dynamicPrecision = (price) => {
    if (price < 0.001) return 8;
    if (price < 0.1) return 6;
    if (price < 1) return 4;
    return 2;
  };
  const formatPrice = (price) => price.toLocaleString(undefined, { minimumFractionDigits: dynamicPrecision(price), maximumFractionDigits: dynamicPrecision(price) });
  const handleTpPercentChange = (val: number) => {
    const fixedVal = Math.abs(Number(val.toFixed(4)));
    setTpPercent(fixedVal);
    if (currentPrice > 0) {
      // TP: Long goes UP (+), Short goes DOWN (-)
      const multiplier = side === 'BUY' ? (1 + fixedVal / 100) : (1 - fixedVal / 100);
      setTpPrice(Number((currentPrice * multiplier).toFixed(dynamicPrecision(currentPrice))));
    }
  };

  const handleSlPercentChange = (val: number) => {
    const fixedVal = Math.abs(Number(val.toFixed(4)));
    setSlPercent(fixedVal);
    if (currentPrice > 0) {
      // SL: Long goes DOWN (-), Short goes UP (+)
      const multiplier = side === 'BUY' ? (1 - fixedVal / 100) : (1 + fixedVal / 100);
      setSlPrice(Number((currentPrice * multiplier).toFixed(dynamicPrecision(currentPrice))));
    }
  };

  const handleTpPriceChange = (val: number) => {
    setTpPrice(val);
    if (currentPrice > 0) {
      // TP: Distance is positive in the profit direction
      const diff = side === 'BUY' ? (val - currentPrice) : (currentPrice - val);
      setTpPercent(Number(Math.abs(diff / currentPrice * 100).toFixed(4)));
    }
  };

  const handleSlPriceChange = (val: number) => {
    setSlPrice(val);
    if (currentPrice > 0) {
      // SL: Distance is positive in the loss direction
      const diff = side === 'BUY' ? (currentPrice - val) : (val - currentPrice);
      setSlPercent(Number(Math.abs(diff / currentPrice * 100).toFixed(4)));
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

  const potentialProfit = useMemo(() => {
    if (!tpEnabled || tpPrice <= 0 || currentPrice <= 0) return 0;
    const diff = side === 'BUY' ? (tpPrice - currentPrice) : (currentPrice - tpPrice);
    return diff * quantity;
  }, [tpEnabled, tpPrice, currentPrice, quantity, side]);

  const potentialLoss = useMemo(() => {
    if (!slEnabled || slPrice <= 0 || currentPrice <= 0) return 0;
    const diff = side === 'BUY' ? (currentPrice - slPrice) : (slPrice - currentPrice);
    return diff * quantity;
  }, [slEnabled, slPrice, currentPrice, quantity, side]);

  // Estimated Liquidation Price (More precise approximation)
  const liqPrice = useMemo(() => {
    if (tradingMode !== 'LEAD' || currentPrice <= 0 || leverage <= 0) return 0;
    
    // Maintenance Margin Rate (MMR) for lower tiers on Binance is typically 0.4% (0.004)
    const mmr = 0.004; 
    
    // Long: Liq = Entry * (1 - (1/Lev) + MMR)
    // Short: Liq = Entry * (1 + (1/Lev) - MMR)
    if (side === 'BUY') {
        return currentPrice * (1 - (1 / leverage) + mmr);
    } else {
        return currentPrice * (1 + (1 / leverage) - mmr);
    }
  }, [currentPrice, leverage, side, tradingMode]);

  const riskBuffer = useMemo(() => {
    if (liqPrice <= 0 || currentPrice <= 0) return 0;
    return Math.abs(((liqPrice - currentPrice) / currentPrice * 100));
  }, [liqPrice, currentPrice]);

  const quoteAsset = symbol.endsWith('USDT') ? 'USDT' : 'USDC';
  const modeColor = tradingMode === 'LEAD' ? 'orange' : 'blue';

  return (
    <div className={`bg-slate-950 p-6 h-full rounded-2xl border ${tradingMode === 'LEAD' ? 'border-orange-900/30 shadow-orange-900/10' : 'border-slate-800'} shadow-xl overflow-y-auto overflow-x-hidden transition-all duration-500 custom-terminal-scrollbar`}>
        <h2 className="text-xl font-black mb-6 tracking-tight flex items-center justify-between text-white uppercase">
            <div className="flex items-center gap-2 text-sm xl:text-lg">
                <div className={`w-2 h-2 bg-${modeColor}-500 rounded-full animate-pulse`}></div>
                <span className="truncate">{tradingMode === 'SPOT' ? 'Smart Buy' : 'Lead Terminal'}</span>
            </div>
            {tradingMode === 'LEAD' && <span className="text-[8px] xl:text-[10px] bg-orange-500/10 text-orange-400 px-2 py-0.5 rounded border border-orange-500/20 shrink-0">FUTURES</span>}
        </h2>
      
      <div className="space-y-5">
        {/* Side Selector (Only for Lead Trading) */}
        {tradingMode === 'LEAD' && (
          <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800">
            <button 
              onClick={() => setSide('BUY')}
              className={`flex-1 py-2 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${side === 'BUY' ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/40' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <TrendingUp size={12} /> Long
            </button>
            <button 
              onClick={() => setSide('SELL')}
              className={`flex-1 py-2 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${side === 'SELL' ? 'bg-rose-600 text-white shadow-lg shadow-rose-900/40' : 'text-slate-500 hover:text-slate-300'}`}
            >
              <TrendingDown size={12} /> Short
            </button>
          </div>
        )}

        {/* Leverage Slider (Only for Lead Trading) */}
        {tradingMode === 'LEAD' && (
            <div className="space-y-2 bg-slate-900/50 p-3 rounded-2xl border border-slate-800 group relative">
                <div className="flex justify-between items-center">
                    <label className="text-[9px] text-slate-500 font-black uppercase tracking-widest">Initial Leverage</label>
                    <span className="text-xs font-mono font-black text-orange-400">{leverage}x</span>
                </div>
                <input 
                    type="range" min="1" max="50" step="1" value={leverage}
                    onChange={(e) => setLeverage(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-500"
                />
                <div className="flex justify-between text-[7px] font-black text-slate-600 uppercase tracking-tighter">
                    <span>1x</span>
                    <span>10x</span>
                    <span>25x</span>
                    <span>50x</span>
                </div>

                {/* Leverage Tooltip */}
                <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-4 w-48 p-3 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none transition-all duration-300 z-50">
                    <p className="text-[9px] font-black text-orange-500 uppercase tracking-widest mb-2 border-b border-slate-800 pb-2">Impact</p>
                    <div className="space-y-1.5">
                        <div className="flex justify-between">
                            <span className="text-[8px] text-slate-500 font-bold uppercase">Power</span>
                            <span className="text-[9px] text-white font-black">{leverage}x</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[8px] text-slate-500 font-bold uppercase">Margin</span>
                            <span className="text-[9px] text-white font-black">{(100 / leverage).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        )}

        <div className="grid grid-cols-1 gap-3">
          <div className="space-y-1.5">
            <label className="text-[9px] text-slate-500 font-black uppercase tracking-widest ml-1">Margin ({quoteAsset})</label>
            <input 
              type="number" value={usdcAmount} onChange={e => handleUsdcChange(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-base font-mono outline-none focus:border-blue-500 transition-all text-white font-black"
            />
          </div>
          <div className="space-y-1.5 relative group">
            <label className="text-[9px] text-slate-500 font-black uppercase tracking-widest ml-1">Size ({symbol.replace(quoteAsset, '')})</label>
            <input 
              type="number" step="0.0001" value={quantity} onChange={e => handleQuantityChange(Number(e.target.value))}
              className="w-full bg-slate-800/30 border border-slate-800 rounded-xl px-4 py-2.5 text-base font-mono outline-none focus:border-blue-500/50 transition-all text-slate-300"
            />
            {leverage > 1 && (
                <div className="absolute right-4 bottom-2.5 text-[8px] font-black text-slate-600 uppercase">
                    Val: ${(quantity * currentPrice).toFixed(2)}
                </div>
            )}
          </div>
        </div>

        {/* Risk Indicators (Futures Only) */}
        {tradingMode === 'LEAD' && liqPrice > 0 && (
            <div className="flex gap-2">
                <div className="flex-1 bg-rose-500/5 border border-rose-500/20 p-2.5 rounded-xl flex flex-col items-center group relative cursor-help min-w-0">
                    <span className="text-[7px] xl:text-[8px] font-black text-rose-500 uppercase mb-0.5 truncate w-full text-center">Liquidation</span>
                    <span className="text-[10px] xl:text-xs font-mono font-black text-white truncate w-full text-center">${formatPrice(liqPrice)}</span>
                    
                    <div className="absolute left-0 right-0 bottom-full mb-4 w-56 p-3 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none transition-all duration-300 z-50 mx-auto">
                        <div className="flex items-center gap-2 mb-2 border-b border-slate-800 pb-2">
                            <ShieldAlert size={12} className="text-rose-500" />
                            <p className="text-[9px] font-black text-rose-500 uppercase tracking-widest">Safety Warning</p>
                        </div>
                        <p className="text-[8px] text-slate-400 leading-tight font-medium">
                            If price hits <span className="text-white font-black">${formatPrice(liqPrice)}</span>, position will be liquidated.
                        </p>
                    </div>
                </div>

                <div className="flex-1 bg-slate-900 border border-slate-800 p-2.5 rounded-xl flex flex-col items-center group relative cursor-help min-w-0">
                    <span className="text-[7px] xl:text-[8px] font-black text-slate-500 uppercase mb-0.5 truncate w-full text-center">Risk Buffer</span>
                    <span className={`text-[10px] xl:text-xs font-mono font-black truncate w-full text-center ${riskBuffer > 10 ? 'text-emerald-400' : riskBuffer > 5 ? 'text-orange-400' : 'text-rose-400'}`}>
                        {riskBuffer.toFixed(2)}%
                    </span>

                    <div className="absolute left-0 right-0 bottom-full mb-4 w-56 p-3 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none transition-all duration-300 z-50 mx-auto">
                        <div className="flex items-center gap-2 mb-2 border-b border-slate-800 pb-2">
                            <Zap size={12} className="text-blue-400" />
                            <p className="text-[9px] font-black text-blue-400 uppercase tracking-widest">Price Gap</p>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-[8px] text-slate-500 font-bold uppercase">Status</span>
                            <span className={`text-[9px] font-black ${riskBuffer > 10 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {riskBuffer > 10 ? 'SAFE' : 'CRITICAL'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        )}

        <div className="space-y-3 pt-4 border-t border-slate-800">
          <div className={`p-3 rounded-2xl border transition-all ${tpEnabled ? 'bg-emerald-900/10 border-emerald-900/30' : 'bg-slate-800/30 border-slate-800 opacity-40'}`}>
            <div className="flex justify-between items-center mb-1.5">
              <div className="flex items-center gap-2">
                <input type="checkbox" checked={tpEnabled} onChange={() => setTpEnabled(!tpEnabled)} className="w-3.5 h-3.5 accent-emerald-500 cursor-pointer" />
                <label className="text-[9px] text-emerald-400 font-black uppercase tracking-widest">Take Profit</label>
              </div>
              {tpEnabled && (
                <div className="flex items-center gap-1 bg-slate-950 rounded-lg border border-emerald-500/20 px-1">
                  <button onClick={() => handleTpPercentChange(tpPercent - 0.1)} className="p-1 text-slate-600 hover:text-emerald-400"><ArrowDown size={10} /></button>
                  <input 
                    type="number" step="0.1" value={tpPercent} 
                    onChange={e => handleTpPercentChange(parseFloat(e.target.value))}
                    className="w-10 bg-transparent text-emerald-400 font-black text-[10px] outline-none text-right"
                  />
                  <span className="text-[9px] text-emerald-500 font-black pr-1">%</span>
                  <button onClick={() => handleTpPercentChange(tpPercent + 0.1)} className="p-1 text-slate-600 hover:text-emerald-400"><ArrowUp size={10} /></button>
                </div>
              )}
            </div>
            <div className="flex justify-between items-end">
                <input 
                    type="number" value={tpPrice} onChange={e => handleTpPriceChange(Number(e.target.value))} disabled={!tpEnabled}
                    className="w-1/2 bg-transparent border-none text-xl font-mono outline-none text-white font-black"
                />
                {tpEnabled && (
                    <div className="text-right flex flex-col items-end pb-1">
                        <span className="text-[7px] text-slate-500 font-black uppercase tracking-tighter">Est. Profit</span>
                        <span className="text-[10px] font-mono font-black text-emerald-400">+{potentialProfit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} {quoteAsset}</span>
                    </div>
                )}
            </div>
          </div>

          <div className={`p-3 rounded-2xl border transition-all ${slEnabled ? 'bg-rose-900/10 border-rose-900/30' : 'bg-slate-800/30 border-slate-800 opacity-40'}`}>
            <div className="flex justify-between items-center mb-1.5">
              <div className="flex items-center gap-2">
                <input type="checkbox" checked={slEnabled} onChange={() => setSlEnabled(!slEnabled)} className="w-3.5 h-3.5 accent-rose-500 cursor-pointer" />
                <label className="text-[9px] text-rose-400 font-black uppercase tracking-widest">Stop Loss</label>
              </div>
              {slEnabled && (
                <div className="flex items-center gap-1 bg-slate-950 rounded-lg border border-rose-500/20 px-1">
                  <button onClick={() => handleSlPercentChange(slPercent - 0.1)} className="p-1 text-slate-600 hover:text-rose-400"><ArrowDown size={10} /></button>
                  <input 
                    type="number" step="0.1" value={slPercent} 
                    onChange={e => handleSlPercentChange(parseFloat(e.target.value))}
                    className="w-10 bg-transparent text-rose-400 font-black text-[10px] outline-none text-right"
                  />
                  <span className="text-[9px] text-rose-500 font-black pr-1">%</span>
                  <button onClick={() => handleSlPercentChange(slPercent + 0.1)} className="p-1 text-slate-600 hover:text-rose-400"><ArrowUp size={10} /></button>
                </div>
              )}
            </div>
            <div className="flex justify-between items-end">
                <input 
                    type="number" value={slPrice} onChange={e => handleSlPriceChange(Number(e.target.value))} disabled={!slEnabled}
                    className="w-1/2 bg-transparent border-none text-xl font-mono outline-none text-white font-black"
                />
                {slEnabled && (
                    <div className="text-right flex flex-col items-end pb-1">
                        <span className="text-[7px] text-slate-500 font-black uppercase tracking-tighter">Est. Loss</span>
                        <span className="text-[10px] font-mono font-black text-rose-400">-{Math.abs(potentialLoss).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} {quoteAsset}</span>
                    </div>
                )}
            </div>
          </div>
        </div>

        <button 
          onClick={onTradeClick}
          disabled={isTrading}
          className={`w-full ${isTrading ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : (tradingMode === 'SPOT' ? 'bg-blue-600 hover:bg-blue-500' : (side === 'BUY' ? 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-900/20' : 'bg-rose-600 hover:bg-rose-500 shadow-rose-900/20'))} text-white font-black py-3.5 rounded-2xl mt-2 transition-all shadow-lg uppercase tracking-widest text-xs flex items-center justify-center gap-3`}
        >
          {isTrading ? (
            <>
              <div className="w-4 h-4 border-2 border-slate-600 border-t-blue-400 rounded-full animate-spin"></div>
              Processing...
            </>
          ) : (
            tradingMode === 'SPOT' ? 'Execute Smart Trade' : `Execute ${side === 'BUY' ? 'Long' : 'Short'}`
          )}
        </button>

        <div className="pt-3 border-t border-slate-800">
            <button
                onClick={() => handleMarketClose()}
                className="w-full bg-slate-800 hover:bg-rose-900/50 hover:text-rose-400 text-slate-400 font-black py-2.5 rounded-xl transition-all text-[9px] uppercase tracking-widest border border-slate-700"
            >
                Market Close ({assetBalance.toFixed(4)})
            </button>
        </div>
      </div>
    </div>
  );
};

export default SmartTerminalView;
