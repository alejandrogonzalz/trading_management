import { useState, useEffect } from 'react';
import TradingChart from './components/TradingChart';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [balances, setBalances] = useState([]);
  const [openOrders, setOpenOrders] = useState([]);
  const [symbol, setSymbol] = useState('BTCUSDC');
  const [quantity, setQuantity] = useState(0.001);
  const [tp, setTp] = useState(70000);
  const [sl, setSl] = useState(65000);

  const fetchStatus = async () => {
    try {
      const bRes = await fetch(`${API_BASE}/account/balances`);
      const bData = await bRes.json();
      setBalances(bData);

      const oRes = await fetch(`${API_BASE}/trades/open`);
      const oData = await oRes.json();
      setOpenOrders(oData);
    } catch (err) {
      console.error("Error fetching status:", err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSmartTrade = async () => {
    try {
      const res = await fetch(`${API_BASE}/trades/smart-trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          side: 'BUY',
          quantity: Number(quantity),
          take_profit_price: Number(tp),
          stop_loss_price: Number(sl)
        })
      });
      const data = await res.json();
      alert(`Smart Trade Created! Entry: ${data.entry.orderId}`);
      fetchStatus();
    } catch (err) {
      alert("Trade failed: " + err.message);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 w-full max-w-7xl mx-auto">
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold text-blue-400">Trading Management (USDC Only)</h1>
        <div className="flex gap-4">
          {balances.map(b => (
            <div key={b.asset} className="bg-slate-900 px-4 py-2 rounded-lg border border-slate-800">
              <span className="text-slate-400 text-sm">{b.asset}</span>
              <p className="font-mono">{Number(b.free).toFixed(2)}</p>
            </div>
          ))}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Chart Column */}
        <div className="lg:col-span-2 space-y-8">
          <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 h-[500px]">
            <TradingChart />
          </div>

          <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
            <h2 className="text-xl font-semibold mb-4">Open USDC Orders</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-800">
                    <th className="py-2">Symbol</th>
                    <th className="py-2">Side</th>
                    <th className="py-2">Price</th>
                    <th className="py-2">Qty</th>
                    <th className="py-2">Status</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-sm">
                  {openOrders.map(o => (
                    <tr key={o.orderId} className="border-b border-slate-800/50">
                      <td className="py-2">{o.symbol}</td>
                      <td className={`py-2 ${o.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>{o.side}</td>
                      <td className="py-2">{Number(o.price).toFixed(2)}</td>
                      <td className="py-2">{o.origQty}</td>
                      <td className="py-2 text-slate-400">{o.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Smart Trade Panel */}
        <div className="space-y-8">
          <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <span className="bg-blue-500 w-2 h-6 rounded-full inline-block"></span>
              Smart Trade
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm text-slate-400 block mb-1">Symbol</label>
                <input 
                  value={symbol} onChange={e => setSymbol(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 focus:border-blue-500 outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-slate-400 block mb-1">Quantity</label>
                  <input 
                    type="number" value={quantity} onChange={e => setQuantity(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 outline-none"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-400 block mb-1">Market Buy</label>
                  <div className="bg-slate-950 border border-slate-800 rounded px-3 py-2 text-slate-500">
                    Instant Fill
                  </div>
                </div>
              </div>

              <div className="border-t border-slate-800 pt-4 mt-4">
                <div className="flex justify-between mb-2">
                  <label className="text-sm text-green-400">Take Profit</label>
                  <span className="text-xs text-slate-500">Limit Sell</span>
                </div>
                <input 
                  type="number" value={tp} onChange={e => setTp(e.target.value)}
                  className="w-full bg-slate-950 border border-green-900/50 rounded px-3 py-2 outline-none"
                />
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm text-red-400">Stop Loss</label>
                  <span className="text-xs text-slate-500">Stop Limit</span>
                </div>
                <input 
                  type="number" value={sl} onChange={e => setSl(e.target.value)}
                  className="w-full bg-slate-950 border border-red-900/50 rounded px-3 py-2 outline-none"
                />
              </div>

              <button 
                onClick={handleSmartTrade}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded-lg mt-6 transition-colors shadow-lg"
              >
                Execute Smart Trade
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
