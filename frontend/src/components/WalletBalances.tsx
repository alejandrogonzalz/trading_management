import { useMemo } from 'react';
import { Wallet, Zap, ShieldCheck, TrendingUp, BarChart2 } from 'lucide-react';

const WalletBalances = ({ balances = [], quoteAsset = 'USDT', globalPrices = [], mode = 'SPOT' }) => {
  const isSpot = mode === 'SPOT';
  
  const quoteBalance = useMemo(() => {
    return Array.isArray(balances) ? balances.find(b => b.asset === quoteAsset) : null;
  }, [balances, quoteAsset]);
  
  const otherBalances = useMemo(() => {
    if (!Array.isArray(balances)) return [];
    
    if (isSpot) {
      return balances
        .filter(b => b.asset !== quoteAsset && (parseFloat(b.free) > 0 || parseFloat(b.locked) > 0))
        .map(b => {
          const symbolStr = `${b.asset}${quoteAsset}`;
          const priceObj = globalPrices.find(p => (p as any).symbol === symbolStr);
          const price = priceObj ? parseFloat((priceObj as any).price) : 0;
          const total = parseFloat(b.free) + parseFloat(b.locked);
          return { ...b, price, value: total * price };
        })
        .sort((a, b) => b.value - a.value);
    } else {
      // Futures v2 structure: assets list
      return balances
        .filter(b => parseFloat(b.walletBalance) > 0)
        .map(b => {
           return {
             asset: b.asset,
             balance: parseFloat(b.walletBalance),
             available: parseFloat(b.availableBalance),
             upnl: parseFloat(b.unrealizedProfit),
             margin: parseFloat(b.marginBalance),
             value: parseFloat(b.marginBalance) // In futures, Margin Balance is the 'liquidation value'
           };
        })
        .sort((a, b) => b.value - a.value);
    }
  }, [balances, globalPrices, quoteAsset, isSpot]);

  const totalPortfolioValue = useMemo(() => {
    if (isSpot) {
      const quoteAmt = quoteBalance ? parseFloat(quoteBalance.free) + parseFloat(quoteBalance.locked) : 0;
      const othersValue = otherBalances.reduce((sum, b) => sum + b.value, 0);
      return quoteAmt + othersValue;
    } else {
      return Array.isArray(balances) ? balances.reduce((sum, b) => sum + parseFloat(b.marginBalance || 0), 0) : 0;
    }
  }, [quoteBalance, otherBalances, balances, isSpot]);

  const totalUnrealizedPnl = useMemo(() => {
    if (isSpot) return 0;
    return Array.isArray(balances) ? balances.reduce((sum, b) => sum + parseFloat(b.unrealizedProfit || 0), 0) : 0;
  }, [balances, isSpot]);

  // UI Helper for Mode colors to avoid purge issues
  const theme = isSpot ? {
      base: 'blue',
      bg: 'bg-blue-600/5',
      border: 'border-blue-500/20',
      hoverBorder: 'group-hover:border-blue-500/50',
      hoverBg: 'group-hover:bg-blue-600/10',
      text: 'text-blue-400',
      accent: 'text-blue-100'
  } : {
      base: 'orange',
      bg: 'bg-orange-600/5',
      border: 'border-orange-500/20',
      hoverBorder: 'group-hover:border-orange-500/50',
      hoverBg: 'group-hover:bg-orange-600/10',
      text: 'text-orange-400',
      accent: 'text-orange-100'
  };

  return (
    <div className="relative group">
      <div className={`${theme.bg} ${theme.border} ${theme.hoverBorder} ${theme.hoverBg} px-6 py-2 rounded-xl border flex flex-col items-end justify-center shadow-lg transition-all cursor-help min-w-[160px] h-[52px]`}>
        <span className={`text-[8px] font-black uppercase tracking-[0.2em] mb-0.5 flex items-center gap-1 ${theme.text}`}>
          <Wallet size={10} /> {isSpot ? 'Spot Portfolio' : 'Futures Portfolio'}
        </span>
        <div className="flex items-baseline gap-2">
            {!isSpot && totalUnrealizedPnl !== 0 && (
                <span className={`text-[10px] font-bold ${totalUnrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {totalUnrealizedPnl >= 0 ? '+' : ''}{totalUnrealizedPnl.toFixed(2)}
                </span>
            )}
            <p className={`font-mono text-lg font-black tracking-tighter ${theme.accent}`}>
            ${totalPortfolioValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
        </div>
      </div>

      {/* MEGA DROP DOWN */}
      <div className="absolute top-full right-0 mt-4 w-[520px] bg-slate-900/98 backdrop-blur-2xl border border-slate-700/50 rounded-3xl shadow-[0_30px_100px_rgba(0,0,0,0.9)] overflow-hidden opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[2000] transform origin-top-right group-hover:translate-y-0 translate-y-4 scale-95 group-hover:scale-100">
        <div className="p-6 border-b border-slate-800 bg-slate-950/50 flex justify-between items-center text-left">
          <h4 className="text-xs font-black text-white uppercase tracking-[0.2em] flex items-center gap-3">
            <div className={`w-2 h-2 rounded-full animate-pulse ${isSpot ? 'bg-blue-500' : 'bg-orange-500'}`}></div>
            {isSpot ? 'Spot Portfolio Breakdown' : 'Futures Account Breakdown'}
          </h4>
          <span className={`text-[10px] font-black uppercase tracking-widest px-3 py-1 rounded-full border ${
              isSpot ? 'text-blue-400 bg-blue-500/10 border-blue-500/20' : 'text-orange-400 bg-orange-500/10 border-orange-500/20'
          }`}>
              {isSpot ? 'Spot Wallet' : 'USDS-M Futures'}
          </span>
        </div>
        
        <div className="p-2">
            <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar relative text-left">
                <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 z-20">
                    <tr className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-900">
                    <th className="px-4 py-3 border-b border-slate-800">Asset</th>
                    <th className="px-4 py-3 text-right border-b border-slate-800">{isSpot ? 'Balance' : 'Wallet'}</th>
                    <th className="px-4 py-3 text-right border-b border-slate-800">{isSpot ? 'Price' : 'UPnL'}</th>
                    <th className="px-4 py-3 text-right border-b border-slate-800">{isSpot ? `Value (${quoteAsset})` : 'Margin Bal.'}</th>
                    </tr>
                </thead>
                <tbody className="font-mono">
                    {isSpot ? (
                        <>
                            {quoteBalance && (
                                <tr className="bg-blue-500/10 transition-colors border-b border-slate-800/50">
                                    <td className="px-4 py-4 text-sm text-white font-black">{quoteAsset}</td>
                                    <td className="px-4 py-4 text-right text-sm text-slate-300">{(parseFloat(quoteBalance.free) + parseFloat(quoteBalance.locked)).toLocaleString()}</td>
                                    <td className="px-4 py-4 text-right text-sm text-slate-600">1.00</td>
                                    <td className="px-4 py-4 text-right text-sm text-emerald-400 font-black">${parseFloat(quoteBalance.free).toLocaleString()}</td>
                                </tr>
                            )}
                            {otherBalances.map(b => (
                                <tr key={b.asset} className="hover:bg-slate-800/50 transition-colors border-b border-slate-800/30 last:border-0">
                                    <td className="px-4 py-4 text-sm text-white font-black">{b.asset}</td>
                                    <td className="px-4 py-4 text-right text-sm text-slate-400">{parseFloat(b.free).toFixed(4)}</td>
                                    <td className="px-4 py-4 text-right text-sm text-slate-500">${b.price < 1 ? b.price.toFixed(6) : b.price.toFixed(2)}</td>
                                    <td className="px-4 py-4 text-right text-sm text-white font-black">${b.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                                </tr>
                            ))}
                        </>
                    ) : (
                        otherBalances.map(b => (
                            <tr key={b.asset} className="hover:bg-slate-800/50 transition-colors border-b border-slate-800/30 last:border-0">
                                <td className="px-4 py-4 text-sm text-white font-black">{b.asset}</td>
                                <td className="px-4 py-4 text-right text-sm text-slate-400">${b.balance.toFixed(2)}</td>
                                <td className={`px-4 py-4 text-right text-sm font-bold ${b.upnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {b.upnl >= 0 ? '+' : ''}{b.upnl.toFixed(2)}
                                </td>
                                <td className="px-4 py-4 text-right text-sm text-orange-400 font-black">${b.margin.toFixed(2)}</td>
                            </tr>
                        ))
                    )}
                </tbody>
                </table>
            </div>
          
          {otherBalances.length === 0 && (
            <div className="py-12 text-center">
              <Zap size={32} className="mx-auto text-slate-800 mb-2" />
              <p className="text-[10px] text-slate-600 font-bold uppercase tracking-widest">No assets detected</p>
            </div>
          )}
        </div>
        
        <div className="bg-slate-950/80 p-4 border-t border-slate-800 flex justify-between items-center text-left">
           <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">
               {isSpot ? 'Live exchange rates from Binance' : 'Real-time Futures Margin Data'}
           </span>
           <div className="flex gap-4">
              <div className="flex flex-col items-end">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">Total Equity</span>
                <span className="text-xs font-black text-slate-300">
                  ${totalPortfolioValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
              {!isSpot && (
                  <div className="flex flex-col items-end">
                    <span className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">Account UPnL</span>
                    <span className={`text-xs font-black ${totalUnrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {totalUnrealizedPnl >= 0 ? '+' : ''}${totalUnrealizedPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </div>
              )}
           </div>
        </div>
      </div>
    </div>
  );
};

export default WalletBalances;
