import { useEffect, useRef } from 'react';
import { createChart, ColorType, LineStyle } from 'lightweight-charts';

// Helper function for EMA
const calculateEMA = (data, period) => {
  const emaArray = [];
  let k = 2 / (period + 1);
  let ema = data[0].close; // Start with the first close
  for (let i = 1; i < data.length; i++) {
    ema = data[i].close * k + ema * (1 - k);
    emaArray.push({ time: data[i].time, value: ema });
  }
  return emaArray;
};

const TradingChart = ({ symbol, interval, plannedTp, plannedSl, openOrders }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const ema20SeriesRef = useRef<any>(null);
  const ema50SeriesRef = useRef<any>(null);

  const priceLinesRef = useRef([]);

  // Function to draw a price line
  const drawLine = (series, price, color, title) => {
    if (price > 0 && series) {
      const line = series.createPriceLine({
        price: price,
        color: color,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: title,
      });
      priceLinesRef.current.push(line);
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
      timeScale: { borderColor: '#1e293b', timeVisible: true, secondsVisible: false }
    });
    chartRef.current = chart;

    candlestickSeriesRef.current = chart.addCandlestickSeries({ upColor: '#4ade80', downColor: '#f87171', borderVisible: false });
    volumeSeriesRef.current = chart.addHistogramSeries({ color: '#3b82f6', priceFormat: { type: 'volume' }, priceScaleId: '' });
    volumeSeriesRef.current.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    ema20SeriesRef.current = chart.addLineSeries({ color: '#3b82f6', lineWidth: 2, crosshairMarkerVisible: false });
    ema50SeriesRef.current = chart.addLineSeries({ color: '#f97316', lineWidth: 2, crosshairMarkerVisible: false });

    const handleResize = () => {
      if (chartContainerRef.current) chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Effect for drawing planned orders
  useEffect(() => {
    if (!candlestickSeriesRef.current) return;
    const series = candlestickSeriesRef.current;
    
    // Clear previous lines by removing them one by one
    priceLinesRef.current.forEach(line => series.removePriceLine(line));
    priceLinesRef.current = [];

    // Draw planned lines
    drawLine(series, plannedTp, '#4ade80', 'TP');
    drawLine(series, plannedSl, '#f87171', 'SL');
    
    // Draw actual open orders
    openOrders.forEach(order => {
        if(order.type === "LIMIT") { // Only draw limit orders
            const title = `${order.side} ${order.origQty}`;
            drawLine(series, parseFloat(order.price), order.side === 'BUY' ? '#a3e635' : '#fb923c', title);
        }
    });

  }, [plannedTp, plannedSl, openOrders, symbol]);

  // Effect for fetching data
  useEffect(() => {
    const fetchKlines = async () => {
      if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;
      try {
        const response = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=200`);
        const data = await response.json();
        
        const candleData = data.map(d => ({ time: d[0] / 1000, open: parseFloat(d[1]), high: parseFloat(d[2]), low: parseFloat(d[3]), close: parseFloat(d[4]) }));
        const volData = data.map(d => ({ time: d[0] / 1000, value: parseFloat(d[5]), color: parseFloat(d[4]) >= parseFloat(d[1]) ? '#4ade8055' : '#f8717155' }));

        candlestickSeriesRef.current.setData(candleData);
        volumeSeriesRef.current.setData(volData);
        
        // Calculate and set EMAs
        if (candleData.length > 50) {
            ema20SeriesRef.current.setData(calculateEMA(candleData, 20));
            ema50SeriesRef.current.setData(calculateEMA(candleData, 50));
        }
        
      } catch (err) { console.error("Error fetching klines:", err); }
    };

    fetchKlines();
    const intervalId = setInterval(fetchKlines, 30000);
    return () => clearInterval(intervalId);
  }, [symbol, interval]);

  return <div ref={chartContainerRef} className="w-full h-full" />;
};

export default TradingChart;
