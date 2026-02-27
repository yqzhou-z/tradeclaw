import json
import os
import ccxt

PORTFOLIO_FILE = "paper_portfolio.json"


def calculate_pnl():
    if not os.path.exists(PORTFOLIO_FILE):
        print("[-] Portfolio file not found! Has the bot started trading yet?")
        return

    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)

    print("\n" + "=" * 55)
    print("💰 QUANT PORTFOLIO LIVE DASHBOARD 💰")
    print("=" * 55 + "\n")

    usdt_balance = portfolio.get("USDT", 0.0)
    holdings = portfolio.get("holdings", {})

    print(f"💵 Available Cash: ${usdt_balance:,.2f} USDT\n")

    if not holdings or all(amount <= 0 for amount in holdings.values()):
        print("📉 Current Holdings: None. Waiting for trading signals...")
        print(f"📊 Total Account Value: ${usdt_balance:,.2f}")
        return

    try:
        # Connecting to Binance.US to fetch real-time prices
        exchange = ccxt.binanceus({'enableRateLimit': True})

        total_crypto_value = 0.0

        # Print Table Header
        print(f"{'COIN':<8} | {'AMOUNT':<12} | {'LIVE PRICE':<14} | {'CURRENT VALUE':<14}")
        print("-" * 55)

        for coin, amount in holdings.items():
            if amount <= 0:
                continue

            symbol = f"{coin}/USDT"
            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                current_value = amount * current_price
                total_crypto_value += current_value

                print(f"{coin:<8} | {amount:<12.6f} | ${current_price:<13.2f} | ${current_value:<13.2f}")
            except Exception as e:
                print(f"{coin:<8} | {amount:<12.6f} | [Error fetching price] | N/A")

        print("-" * 55)

        # Calculate Total PnL (Assuming $10,000 was the starting balance)
        total_account_value = usdt_balance + total_crypto_value
        initial_value = 10000.0
        pnl_usd = total_account_value - initial_value
        pnl_pct = (pnl_usd / initial_value) * 100

        print(f"\n📈 Crypto Asset Value: ${total_crypto_value:,.2f}")
        print(f"🏦 Total Account Value: ${total_account_value:,.2f}")

        if pnl_usd >= 0:
            print(f"🟩 All-Time PnL: +${pnl_usd:,.2f} (+{pnl_pct:.2f}%)")
        else:
            print(f"🟥 All-Time PnL: -${abs(pnl_usd):,.2f} ({pnl_pct:.2f}%)")

    except Exception as e:
        print(f"[-] Error connecting to exchange: {e}")

    print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    print("Fetching live market data to calculate PnL...")
    calculate_pnl()