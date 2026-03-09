import { useState } from 'react';
import { TradingChart } from './components/TradingChart';
import { Activity, TrendingUp, BarChart3, Settings, Shield } from 'lucide-react';
import './App.css';

// Mock initial data
const initialData = [
  { time: '2024-03-01', open: 61000, high: 62500, low: 60500, close: 62000 },
  { time: '2024-03-02', open: 62000, high: 63800, low: 61800, close: 63200 },
  { time: '2024-03-03', open: 63200, high: 64200, low: 62900, close: 63100 },
  { time: '2024-03-04', open: 63100, high: 65000, low: 63000, close: 64800 },
  { time: '2024-03-05', open: 64800, high: 67500, low: 64500, close: 67200 },
  { time: '2024-03-06', open: 67200, high: 68100, low: 66500, close: 67500 },
  { time: '2024-03-07', open: 67500, high: 69200, low: 67300, close: 68900 },
  { time: '2024-03-08', open: 68900, high: 70500, low: 68500, close: 70100 },
  { time: '2024-03-09', open: 70100, high: 72000, low: 69800, close: 71500 },
];

function App() {
  const [data] = useState(initialData);

  return (
    <div className="dashboard">
      <header className="header">
        <div className="logo">TRADING AGENT</div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '15px' }}>
          <Shield size={20} color="#22c55e" />
          <Settings size={20} />
        </div>
      </header>

      <div className="stats-bar">
        <div className="price-stat">
          <span className="stat-label">SYMBOL</span>
          <span className="stat-value">BTC / USDT</span>
        </div>
        <div className="price-stat">
          <span className="stat-label">LAST PRICE</span>
          <span className="stat-value" style={{ color: '#22c55e' }}>$71,500.00</span>
        </div>
        <div className="price-stat">
          <span className="stat-label">24H CHANGE</span>
          <span className="stat-value" style={{ color: '#22c55e' }}>+2.45%</span>
        </div>
        <div className="price-stat">
          <span className="stat-label">24H HIGH</span>
          <span className="stat-value">$72,000.00</span>
        </div>
        <div className="price-stat">
          <span className="stat-label">24H LOW</span>
          <span className="stat-value">$69,800.00</span>
        </div>
      </div>

      <main className="chart-container">
        <TradingChart 
          data={data} 
          colors={{ 
            backgroundColor: '#0f172a',
            textColor: '#94a3b8'
          }} 
        />
      </main>

      <aside className="sidebar">
        <h3>Signals & Indicators</h3>
        
        <div className="indicator-card">
          <div className="indicator-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={16} /> RSI (14)
            </span>
            <span className="stat-value">68.5</span>
          </div>
          <div className="indicator-header">
            <span>Signal:</span>
            <span className="signal-hold">NEUTRAL</span>
          </div>
        </div>

        <div className="indicator-card">
          <div className="indicator-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={16} /> MACD Strategy
            </span>
          </div>
          <div className="indicator-header">
            <span>Signal:</span>
            <span className="signal-buy">STRONG BUY</span>
          </div>
        </div>

        <div className="indicator-card">
          <div className="indicator-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BarChart3 size={16} /> Volume Trend
            </span>
          </div>
          <div className="indicator-header">
            <span>Status:</span>
            <span style={{ color: '#22c55e' }}>INCREASING</span>
          </div>
        </div>

        <div style={{ marginTop: 'auto' }}>
          <button style={{ width: '100%', marginBottom: '10px', background: '#22c55e', color: 'white', border: 'none' }}>
            AUTO-TRADE: ON
          </button>
          <p style={{ fontSize: '0.7rem', color: '#94a3b8', textAlign: 'center' }}>
            Agent is monitoring market conditions...
          </p>
        </div>
      </aside>
    </div>
  );
}

export default App;
