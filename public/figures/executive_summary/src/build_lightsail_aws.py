#!/usr/bin/env python3
"""Genera fig_lightsail_aws.svg incrustando los iconos oficiales de icons/ como
data-URIs base64 (para que el SVG sea un único archivo portable).

Uso (desde public/figures/executive_summary/src/):
    python build_lightsail_aws.py

Luego renderiza a PNG a mano (ver README.md). Las otras dos figuras de Avance 7
(fig_roadmap.svg, fig_before_after.svg) son SVG hechos a mano: se editan
directamente, no las genera este script.
"""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ICONS = os.path.join(HERE, "icons")


def data_uri(filename: str, mime: str) -> str:
    with open(os.path.join(ICONS, filename), "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


LIGHTSAIL_URI = data_uri("aws_lightsail.svg", "image/svg+xml")
S3_URI = data_uri("aws_s3.svg", "image/svg+xml")
BINANCE_URI = data_uri("binance_mark.png", "image/png")

SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 800" font-family="Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#232F3E"/>
    </marker>
    <filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="2.5" stdDeviation="3.5" flood-color="#16191F" flood-opacity="0.20"/>
    </filter>
  </defs>

  <rect x="0" y="0" width="1060" height="800" fill="#FFFFFF"/>

  <!-- ============ AWS CLOUD BOUNDARY ============ -->
  <rect x="318" y="60" width="714" height="312" rx="14" fill="#F2F8FF" stroke="#232F3E" stroke-width="2" stroke-dasharray="8,5"/>
  <text x="338" y="90" font-size="17" font-weight="bold" fill="#232F3E">AWS (regi&#243;n)</text>

  <!-- ============ USUARIO / FRONTEND (fuera de AWS) ============ -->
  <g>
    <rect x="20" y="120" width="252" height="220" rx="12" fill="#FFFFFF" stroke="#545B64" stroke-width="2" filter="url(#softShadow)"/>
    <rect x="64" y="150" width="166" height="104" rx="6" fill="#FFFFFF" stroke="#545B64" stroke-width="2"/>
    <rect x="64" y="150" width="166" height="22" rx="6" fill="#545B64"/>
    <circle cx="78" cy="161" r="3.2" fill="#FFFFFF"/>
    <circle cx="90" cy="161" r="3.2" fill="#FFFFFF"/>
    <circle cx="102" cy="161" r="3.2" fill="#FFFFFF"/>
    <line x1="80" y1="194" x2="214" y2="194" stroke="#A6B0B8" stroke-width="4.5"/>
    <line x1="80" y1="212" x2="182" y2="212" stroke="#A6B0B8" stroke-width="4.5"/>
    <line x1="80" y1="230" x2="198" y2="230" stroke="#A6B0B8" stroke-width="4.5"/>
    <text x="146" y="290" font-size="16.5" font-weight="bold" text-anchor="middle" fill="#16191F">Usuario / Frontend</text>
    <text x="146" y="312" font-size="13" text-anchor="middle" fill="#545B64">(React, fuera de AWS)</text>
  </g>

  <!-- ============ AWS LIGHTSAIL (app server) ============ -->
  <g>
    <rect x="338" y="100" width="278" height="234" rx="12" fill="#FFF6EC" stroke="#FF9900" stroke-width="2.4" filter="url(#softShadow)"/>
    <image href="{LIGHTSAIL_URI}" x="356" y="116" width="52" height="52"/>
    <text x="420" y="138" font-size="16" font-weight="bold" fill="#16191F">AWS Lightsail</text>
    <text x="420" y="157" font-size="12.5" fill="#545B64">Servidor de aplicaci&#243;n (sin GPU)</text>
    <line x1="356" y1="178" x2="598" y2="178" stroke="#FFD9A0" stroke-width="1.4"/>
    <text x="356" y="202" font-size="13" fill="#16191F">&#8226; FastAPI backend (REST)</text>
    <text x="356" y="226" font-size="13" fill="#16191F">&#8226; Agente LangGraph</text>
    <text x="356" y="250" font-size="13" fill="#16191F">&#8226; SQLite + scheduler 30s</text>
    <text x="356" y="276" font-size="13" font-weight="bold" fill="#B7790A">&#8226; IP p&#250;blica est&#225;tica &#9733;</text>
    <text x="372" y="296" font-size="11" fill="#8A5A00">(requisito para Binance, ver nota)</text>
  </g>

  <!-- ============ AMAZON S3 (model storage) ============ -->
  <g>
    <rect x="700" y="100" width="308" height="234" rx="12" fill="#F2FBF4" stroke="#1B8A4B" stroke-width="2.2" filter="url(#softShadow)"/>
    <image href="{S3_URI}" x="718" y="116" width="52" height="52"/>
    <text x="782" y="138" font-size="16" font-weight="bold" fill="#16191F">Amazon S3</text>
    <text x="782" y="157" font-size="12.5" fill="#545B64">Artefactos del modelo (DVC)</text>
    <line x1="718" y1="178" x2="990" y2="178" stroke="#BCE8CB" stroke-width="1.4"/>
    <text x="718" y="202" font-size="13" fill="#16191F">&#8226; Adaptadores QLoRA</text>
    <text x="718" y="226" font-size="13" fill="#16191F">&#8226; GGUF cuantizado</text>
    <text x="718" y="250" font-size="13" fill="#16191F">&#8226; dataset.jsonl</text>
    <text x="718" y="274" font-size="13" fill="#16191F">&#8226; ~20&#8211;25GB versionados</text>
  </g>

  <!-- ============ SERVICIO DE INFERENCIA LLM (fuera de AWS, delegado) ============ -->
  <g>
    <rect x="338" y="470" width="298" height="220" rx="12" fill="#F5F1FC" stroke="#5D408C" stroke-width="2.2" filter="url(#softShadow)"/>
    <rect x="358" y="488" width="48" height="48" rx="6" fill="#5D408C"/>
    <rect x="369" y="499" width="26" height="26" rx="2" fill="#FFFFFF"/>
    <line x1="363" y1="484" x2="363" y2="476" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="376" y1="484" x2="376" y2="476" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="389" y1="484" x2="389" y2="476" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="402" y1="484" x2="402" y2="476" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="363" y1="540" x2="363" y2="548" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="376" y1="540" x2="376" y2="548" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="389" y1="540" x2="389" y2="548" stroke="#5D408C" stroke-width="2.5"/>
    <line x1="402" y1="540" x2="402" y2="548" stroke="#5D408C" stroke-width="2.5"/>
    <text x="420" y="508" font-size="15" font-weight="bold" fill="#16191F">Servicio de inferencia LLM</text>
    <text x="420" y="527" font-size="12.5" fill="#5D408C" font-weight="bold">Together AI o Ollama local</text>
    <line x1="356" y1="548" x2="618" y2="548" stroke="#E0D6F2" stroke-width="1.4"/>
    <text x="356" y="574" font-size="13" fill="#16191F">&#8226; Sirve el modelo ajustado (QLoRA)</text>
    <text x="356" y="600" font-size="13" fill="#16191F">&#8226; Escala independiente del backend</text>
    <text x="356" y="626" font-size="13" fill="#16191F">&#8226; Sin GPU dentro de AWS</text>
  </g>

  <!-- ============ BINANCE API (exchange externo) ============ -->
  <g>
    <rect x="700" y="470" width="308" height="220" rx="12" fill="#FFF3F1" stroke="#C9462C" stroke-width="2.2" filter="url(#softShadow)"/>
    <image href="{BINANCE_URI}" x="718" y="488" width="48" height="48"/>
    <text x="778" y="508" font-size="16" font-weight="bold" fill="#16191F">Binance API</text>
    <text x="778" y="527" font-size="12.5" fill="#545B64">Exchange externo</text>
    <line x1="716" y1="548" x2="990" y2="548" stroke="#F6CFC4" stroke-width="1.4"/>
    <text x="718" y="574" font-size="13" fill="#16191F">&#8226; Velas OHLCV (klines)</text>
    <text x="718" y="600" font-size="13" fill="#16191F">&#8226; &#211;rdenes spot / futuros</text>
    <text x="718" y="626" font-size="13" font-weight="bold" fill="#9E2A2B">&#8226; Whitelist por IP &#9733;</text>
    <text x="734" y="646" font-size="11" fill="#7A2020">(la IP fija de Lightsail)</text>
  </g>

  <!-- ============ ARROWS ============ -->
  <!-- usuario -> lightsail -->
  <line x1="274" y1="238" x2="336" y2="238" stroke="#232F3E" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="266" y="206" width="80" height="20" rx="5" fill="#FFFFFF" stroke="#D6DBDF" stroke-width="0.8"/>
  <text x="306" y="220" font-size="11" font-weight="bold" text-anchor="middle" fill="#232F3E">HTTPS / REST</text>

  <!-- lightsail <-> s3 -->
  <line x1="616" y1="150" x2="698" y2="150" stroke="#232F3E" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="657" y="140" font-size="12" text-anchor="middle" fill="#232F3E">dvc pull</text>

  <!-- lightsail -> servicio inferencia (delegacion, cruza el perimetro AWS) -->
  <line x1="477" y1="334" x2="477" y2="468" stroke="#232F3E" stroke-width="2.4" marker-end="url(#arrow)"/>
  <rect x="372" y="392" width="210" height="24" fill="#FFFFFF" opacity="0.92"/>
  <text x="378" y="409" font-size="12" fill="#232F3E">inferencia delegada (HTTP)</text>

  <!-- lightsail -> binance -->
  <line x1="616" y1="300" x2="760" y2="470" stroke="#232F3E" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="648" y="404" width="132" height="22" fill="#FFFFFF" opacity="0.92"/>
  <text x="654" y="420" font-size="12" fill="#232F3E">klines / &#243;rdenes</text>

  <!-- ============ NOTA EXPLICATIVA (footer) ============ -->
  <rect x="20" y="724" width="1020" height="58" rx="8" fill="#FFFBEA" stroke="#B7790A" stroke-width="1.2"/>
  <text x="40" y="747" font-size="13" fill="#16191F"><tspan font-weight="bold">&#9733; &#191;Por qu&#233; Lightsail y no un servicio serverless?</tspan> Lightsail asigna una IP p&#250;blica</text>
  <text x="40" y="767" font-size="13" fill="#16191F">est&#225;tica al servidor de aplicaci&#243;n &#8212; Binance exige whitelistear esa IP por llave de API; un entorno serverless (IP ef&#237;mera) no lo permite.</text>

</svg>
'''

out_path = os.path.join(HERE, "fig_lightsail_aws.svg")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(SVG)
print(f"written {out_path} ({len(SVG)} bytes)")
