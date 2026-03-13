import { useMemo } from 'react';
import { Wallet, Zap } from 'lucide-react';

const WalletBalances = ({ balances = [], quoteAsset = 'USDT', globalPrices = [] }) => {
  const quoteBalance = Array.isArray(balances) ? balances.find(b => b.asset === quoteAsset) : null;
  
  const otherBalances = useMemo(() => {
    if (!Array.isArray(balances)) return [];
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
        <div className="p-6 border-b border-slate-800 bg-slate-950/50 flex justify-between items-center text-left">
          <h4 className="text-xs font-black text-white uppercase tracking-[0.2em] flex items-center gap-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            Portfolio Breakdown
          </h4>
          <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest bg-slate-800 px-3 py-1 rounded-full border border-slate-700">Spot Wallet</span>
        </div>
        
        <div className="p-2">
          <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar relative text-left">
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
        
        <div className="bg-slate-950/80 p-4 border-t border-slate-800 flex justify-between items-center text-left">
           <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Live exchange rates from Binance</span>
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

export default WalletBalances;
