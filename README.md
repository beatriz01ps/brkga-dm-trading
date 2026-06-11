# brkga-dm-trading
Código e resultados do trabalho de disciplina "Hibridizações BRKGA-DM Aplicadas à Otimização de Estratégias de Trading"


## Estrutura

- `dados/ETHUSDT-5m.csv` — série histórica ETH/USDT em candles de 5 min (Binance)
- `codigo/baseline/` — BRKGA puro (sem DM)
- `codigo/h1-avg/` até `codigo/h6-qx/` — variantes BRKGA-DM (Avg, Std, X, MX, RX, QX)

Cada pasta de variante tem `src/` (Rust), `resultado-e1.log` (expressão original), `resultado-e2.log` (confirmação dupla) e `results/generations.csv` com métricas por geração. O `src/` usa a expressão E2; a E1 está comentada logo acima no `src/backtest/trade_rule.rs`.

## Como rodar

Requer Rust 1.56+. Para compilar e rodar uma variante:

cd codigo/h1-avg
cargo build --release
./target/release/trade-optimizer


O caminho do CSV está em `src/main.rs`. Ajustar se necessário.
