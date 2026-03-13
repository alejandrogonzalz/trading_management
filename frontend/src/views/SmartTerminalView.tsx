import { useState, useEffect } from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

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

export default SmartTerminalView;
