import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, LineStyle } from 'lightweight-charts';
import { Settings2, Eye, EyeOff } from 'lucide-react';

// Robust Helper function for EMA
const calculateEMA = (data, period) => {
  if (!data || data.length < period || period < 1) return [];
  
  const emaArray = [];
  const k = 2 / (period + 1);
  
  // Find first valid close to initialize
  let ema = 0;
  let firstValidIndex = -1;
  for (let i = 0; i < data.length; i++) {
    if (data[i] && typeof data[i].close === 'number' && !isNaN(data[i].close)) {
      ema = data[i].close;
      firstValidIndex = i;
      break;
    }
  }

  if (firstValidIndex === -1) return [];

  for (let i = firstValidIndex; i < data.length; i++) {
    const item = data[i];
    if (!item || typeof item.close !== 'number' || isNaN(item.close)) continue;
    
    // Standard EMA formula
    ema = item.close * k + ema * (1 - k);
    
    // Only push after we have enough data points for the period
    if (i >= firstValidIndex + period - 1) {
      if (item.time) {
        emaArray.push({ time: item.time, value: ema });
      }
    }
  }
  return emaArray;
};

const EMAOverlay = ({ settings, onUpdate }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="absolute top-4 left-4 z-50 flex flex-col gap-2">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-lg text-slate-400 hover:text-white transition-all shadow-xl"
        title="Indicator Settings"
      >
        <Settings2 size={16} />
      </button>

      {isOpen && (
        <div className="p-3 bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-xl shadow-2xl min-w-[180px] animate-in fade-in slide-in-from-top-2 duration-200">
          <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 border-b border-slate-800 pb-2">Indicators</h4>
          <div className="space-y-3">
            {settings.map((ema, idx) => (
              <div key={ema.id} className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => {
                      const newSettings = [...settings];
                      newSettings[idx].enabled = !newSettings[idx].enabled;
                      onUpdate(newSettings);
                    }}
                    className={`transition-colors ${ema.enabled ? 'text-blue-400' : 'text-slate-600'}`}
                  >
                    {ema.enabled ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                  <span className="text-[10px] font-bold text-slate-400">EMA</span>
                  <input 
                    type="number" 
                    value={ema.period}
                    min="1"
                    max="500"
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      if (isNaN(val)) return;
                      const newSettings = [...settings];
                      newSettings[idx].period = Math.max(1, val);
                      onUpdate(newSettings);
                    }}
                    className="w-10 bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-[10px] font-mono text-white outline-none focus:border-blue-500"
                  />
                </div>
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ema.color }}></div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const TradingChart = ({ symbol, interval, plannedTp, plannedSl, openOrders, showTargets, emaSettings, onEmaUpdate }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const emaSeriesRefs = useRef<any>([]);
  const priceLinesRef = useRef([]);
  const [rawKlines, setRawKlines] = useState([]);

  // Function to draw a price line
  const drawLine = (series, price, color, title) => {
    if (price > 0 && series) {
      try {
        const line = series.createPriceLine({
          price: price,
          color: color,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: title,
        });
        priceLinesRef.current.push(line);
      } catch (e) {}
    }
  };
  
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      timeScale: { borderColor: '#1e293b', timeVisible: true, secondsVisible: false },
      crosshair: {
        mode: 0,
        vertLine: { labelBackgroundColor: '#1e293b' },
        horzLine: { labelBackgroundColor: '#1e293b' },
      },
    });
    chartRef.current = chart;

    candlestickSeriesRef.current = chart.addCandlestickSeries({ 
        upColor: '#4ade80', downColor: '#f87171', borderVisible: false,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });
    
    volumeSeriesRef.current = chart.addHistogramSeries({ color: '#3b82f6', priceFormat: { type: 'volume' }, priceScaleId: '' });
    volumeSeriesRef.current.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, []);

  // Sync EMA Series with Settings
  useEffect(() => {
    if (!chartRef.current) return;

    // Correctly clear old EMA series instances
    emaSeriesRefs.current.forEach(item => {
      if (item && item.series) {
        try { chartRef.current.removeSeries(item.series); } catch (e) {}
      }
    });
    emaSeriesRefs.current = [];

    // Add new ones based on settings
    emaSettings.forEach(ema => {
      if (ema.enabled) {
        const series = chartRef.current.addLineSeries({
          color: ema.color,
          lineWidth: 2,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false,
        });
        emaSeriesRefs.current.push({ id: ema.id, series, period: ema.period });
      }
    });

    // Re-trigger data calculation if we have raw klines
    if (rawKlines.length > 0) {
      emaSeriesRefs.current.forEach(item => {
        const emaData = calculateEMA(rawKlines, item.period);
        if (emaData && emaData.length > 0) {
          try { item.series.setData(emaData); } catch (e) { console.error("Error setting EMA data:", e); }
        }
      });
    }
  }, [emaSettings, rawKlines]);

  // Effect for drawing planned orders
  useEffect(() => {
    if (!candlestickSeriesRef.current) return;
    const series = candlestickSeriesRef.current;
    
    // Clear old lines
    priceLinesRef.current.forEach(line => {
      try { series.removePriceLine(line); } catch (e) {}
    });
    priceLinesRef.current = [];

    if (showTargets) {
      drawLine(series, plannedTp, '#4ade80', 'TP');
      drawLine(series, plannedSl, '#f87171', 'SL');
      openOrders.forEach(order => {
          if(order.type === "LIMIT" || order.type === "LIMIT_MAKER" || order.type === "STOP_LOSS_LIMIT") {
              const title = `${order.side} ${order.origQty}`;
              drawLine(series, parseFloat(order.price || order.stopPrice), order.side === 'BUY' ? '#a3e635' : '#fb923c', title);
          }
      });
    }
  }, [plannedTp, plannedSl, openOrders, symbol, showTargets]);

  // Effect for fetching data
  useEffect(() => {
    const fetchKlines = async () => {
      if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;
      try {
        const response = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=200`);
        const data = await response.json();
        if (!data || !Array.isArray(data) || data.length === 0) return;

        const candleData = data.map(d => ({ 
          time: d[0] / 1000, 
          open: parseFloat(d[1]), 
          high: parseFloat(d[2]), 
          low: parseFloat(d[3]), 
          close: parseFloat(d[4]) 
        }));
        
        const volData = data.map(d => ({ 
          time: d[0] / 1000, 
          value: parseFloat(d[5]), 
          color: parseFloat(d[4]) >= parseFloat(d[1]) ? '#4ade8055' : '#f8717155' 
        }));
        
        setRawKlines(candleData);

        const samplePrice = candleData[candleData.length - 1].close;
        let precision = 2;
        let minMove = 0.01;
        if (samplePrice < 0.001) { precision = 8; minMove = 0.00000001; }
        else if (samplePrice < 0.1) { precision = 6; minMove = 0.000001; }
        else if (samplePrice < 1) { precision = 4; minMove = 0.0001; }

        candlestickSeriesRef.current.applyOptions({ priceFormat: { type: 'price', precision, minMove } });
        candlestickSeriesRef.current.setData(candleData);
        volumeSeriesRef.current.setData(volData);
      } catch (err) { console.error("Error fetching klines:", err); }
    };

    fetchKlines();
    const intervalId = setInterval(fetchKlines, 30000);
    return () => clearInterval(intervalId);
  }, [symbol, interval]);

  return (
    <div className="w-full h-full relative group">
      <EMAOverlay settings={emaSettings} onUpdate={onEmaUpdate} />
      <div className="w-full h-full" ref={chartContainerRef} />
    </div>
  );
};

export default TradingChart;
