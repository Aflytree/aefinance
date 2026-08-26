import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import time
import util
import backtest_strategy
import efi_email

LHB30_CHECKPOINT = Path(__file__).resolve().parent / "lhb30_checkpoint.json"

def _load_lhb_checkpoint():
    if not LHB30_CHECKPOINT.exists():
        return {"done": {}, "failed": []}
    try:
        with LHB30_CHECKPOINT.open(encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("done", {})
        data.setdefault("failed", [])
        return data
    except Exception as e:
        logging.warning(f"读取断点文件失败，将重新开始: {e}")
        return {"done": {}, "failed": []}


def _save_lhb_checkpoint(checkpoint):
    tmp = LHB30_CHECKPOINT.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    tmp.replace(LHB30_CHECKPOINT)


def _summarize_backtest(results):
    last_buy = _last_buy_date(results)
    trades = results.get("trades") or []
    holding = bool(trades) and trades[-1].get("type") == "buy"
    return {
        "stock_code": results.get("stock_code"),
        "stock_name": results.get("stock_name"),
        "win_rate": results.get("win_rate"),
        "total_return": results.get("total_return"),
        "annual_return": results.get("annual_return"),
        "sharpe_ratio": results.get("sharpe_ratio"),
        "number_of_trades": results.get("number_of_trades"),
        "initial_capital": results.get("initial_capital"),
        "final_capital": results.get("final_capital"),
        "position": results.get("position"),
        "last_buy_date": last_buy.isoformat() if last_buy else None,
        "is_holding": holding,
    }


def backtest_recent_dragon_stocks(total_lhb_days = 170, single_stock_start_date = '20240323',
                                  win_rate_th = 0.60,
                                  total_return = 2.00):
    """
    最近 N 个交易日龙虎榜去重回测，支持断点续跑。
    """
    try:
        stock_history = __import__("stock_history")
        stock_history._tickplus_disabled = True
        stock_history.BULK_DATA_MODE = True
    except Exception:
        pass
    logging.info(f"开始回测最近{total_lhb_days}个交易日龙虎榜去重股票")
    codes_path = Path(__file__).resolve().parent / "lhb30_codes.json"
    if codes_path.exists():
        with codes_path.open(encoding="utf-8") as f:
            stock_codes = json.load(f)
        logging.info(f"使用已缓存龙虎榜股票列表: {len(stock_codes)}")
    else:
        stock_codes = util.get_recent_days_lhb_stocks(days=total_lhb_days)
        stock_codes = sorted(set(stock_codes))
        with codes_path.open("w", encoding="utf-8") as f:
            json.dump(stock_codes, f, ensure_ascii=False, indent=2)
        logging.info(f"龙虎榜列表已缓存到 {codes_path}")
    stock_codes = sorted(set(stock_codes))
    logging.info(f"去重后股票数: {len(stock_codes)}")

    checkpoint = _load_lhb_checkpoint()
    done = checkpoint["done"]
    failed = set(checkpoint.get("failed") or [])
    logging.info(f"断点已完成 {len(done)} 只，失败 {len(failed)} 只")

    successful_tests = 0
    for i, code in enumerate(stock_codes, 1):
        if code in done or code in failed:
            logging.info(f"[{i}/{len(stock_codes)}] {code} 已处理，跳过")
            if code in done:
                successful_tests += 1
            continue
        try:
            logging.info(f"[{i}/{len(stock_codes)}] 回测 {code}")
            results = backtest_strategy.backtest_strategy(
                code,
                bg=single_stock_start_date,
                initial_capital_=1000000,
                target_return_=0.11,
                stop_loss_=-0.03,
                init_stop_n_times=0,
            )
            if results is None:
                logging.info(f"股票 {code} 回测无结果，可能数据不足")
                failed.add(code)
            else:
                done[code] = _summarize_backtest(results)
                successful_tests += 1
                logging.info(
                    f"完成 {code}  收益 {results.get('total_return', 0):.2%} "
                    f"胜率 {results.get('win_rate', 0):.2%}"
                )
        except Exception as e:
            logging.error(f"回测股票 {code} 时出错: {str(e)}")
            failed.add(code)
        checkpoint["done"] = done
        checkpoint["failed"] = sorted(failed)
        _save_lhb_checkpoint(checkpoint)

    all_stock_results = list(done.values())
    summary_stats = [{
        "date": datetime.now().strftime("%Y%m%d"),
        "stock_count": len(stock_codes),
        "successful_backtests": successful_tests,
        "results": all_stock_results,
        "trades": [],
        "last_buys": [],
    }]
    generate_monthly_report(
        summary_stats,
        len(stock_codes),
        successful_tests,
        all_stock_results,
        win_rate_th,
        total_return,
    )
    return all_stock_results


def generate_monthly_report(summary_stats, total_stocks, successful_tests, all_stock_results,
                            win_rate_th = 0.60,
                            total_return = 2.00):
    """
    生成月度回测汇总报告
    """
    logging.info(f"\n{'=' * 80}")
    logging.info("📊 最近龙虎榜股票回测汇总报告")
    logging.info(f"{'=' * 80}")

    total_days = len(summary_stats)
    total_stocks_attempted = total_stocks
    total_successful = successful_tests

    logging.info(f"回测天数: {total_days}")
    logging.info(f"总龙虎榜股票数: {total_stocks_attempted}")
    logging.info(f"成功回测股票数: {total_successful}")
    logging.info(
        f"回测成功率: {total_successful / total_stocks_attempted * 100:.2f}%" if total_stocks_attempted > 0 else "回测成功率: 0%")

    # 按日期统计
    logging.info(f"\n按日期统计:")
    for stat in summary_stats:
        success_rate = stat['successful_backtests'] / stat['stock_count'] * 100 if stat['stock_count'] > 0 else 0
        logging.info(f"日期 {stat['date']}: {stat['successful_backtests']}/{stat['stock_count']} ({success_rate:.1f}%)")

    # 立即进行高性能股票筛选和展示
    if all_stock_results:
        logging.info(f"\n开始筛选高性能股票...")
        high_performance_stocks = filter_high_performance_stocks(all_stock_results, win_rate_th, total_return)
        for stock in high_performance_stocks:
            if not stock.get("stock_name"):
                stock["stock_name"] = util.get_stock_name(stock.get("stock_code")) or ""
        print_high_performance_details(high_performance_stocks, win_rate_th, total_return)
        generate_performance_report(high_performance_stocks)
        save_high_performance_stocks(high_performance_stocks, win_rate_th, total_return)
        recent_buy_stocks = filter_recent_buy_stocks(high_performance_stocks)
        print_recent_buy_stocks(recent_buy_stocks)
        notify_recent_buy_stocks(recent_buy_stocks, win_rate_th, total_return)
    else:
        logging.info("没有回测结果可供筛选")

    # 保存详细结果到文件（可选）
    save_detailed_results(all_stock_results)


def save_detailed_results(all_stock_results):
    """
    保存详细回测结果到文件
    """
    try:
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dragon_stocks_backtest_{timestamp}.json"

        # 转换结果为可序列化的格式
        serializable_results = []
        for stock_result in all_stock_results:
            # 只保存关键信息，避免数据过大
            serializable_stock = {
                'stock_code': stock_result.get('stock_code'),
                'stock_name': stock_result.get('stock_name'),
                'win_rate': stock_result.get('win_rate'),
                'total_return': stock_result.get('total_return'),
                'annual_return': stock_result.get('annual_return'),
                'sharpe_ratio': stock_result.get('sharpe_ratio'),
                'number_of_trades': stock_result.get('number_of_trades'),
                'initial_capital': stock_result.get('initial_capital'),
                'final_capital': stock_result.get('final_capital')
            }
            serializable_results.append(serializable_stock)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logging.info(f"详细结果已保存到: {filename}")

    except Exception as e:
        logging.error(f"保存结果文件时出错: {str(e)}")


def main():
    """
    主函数
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dragon_stocks_backtest.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    try:
        # 执行回测
        results = backtest_recent_dragon_stocks()

        logging.info("\n🎉 最近龙虎榜股票回测完成！")
        return results

    except Exception as e:
        logging.error(f"回测执行过程中发生错误: {str(e)}")
        return None


def filter_high_performance_stocks(all_stock_results, win_rate_th = 0.60, total_return = 2.00):
    """
    筛选胜率性能股票
    """
    high_performance_stocks = []

    for stock_result in all_stock_results:
        try:
            # 检查必要字段是否存在
            if not all(key in stock_result for key in ['win_rate', 'total_return', 'stock_code']):
                continue

            win_rate = stock_result['win_rate']
            total_return_ = stock_result['total_return']

            # 筛选条件
            if win_rate >= win_rate_th and total_return_ >= total_return:
                # import pdb;pdb.set_trace()
                high_performance_stocks.append(stock_result)

        except (KeyError, TypeError, ValueError) as e:
            logging.warning(f"处理股票 {stock_result.get('stock_code', '未知')} 数据时出错: {e}")
            continue

    return high_performance_stocks


RECENT_BUY_DAYS = 10


def _last_buy_date(stock_result):
    trades = stock_result.get("trades") or []
    buy_trades = [t for t in trades if t.get("type") == "buy"]
    if not buy_trades:
        return None
    return pd.to_datetime(buy_trades[-1]["date"]).date()


def filter_recent_buy_stocks(stock_results, recent_days=RECENT_BUY_DAYS):
    """筛选当前仍持仓，或最近 recent_days 天内有过买入的股票。"""
    today = datetime.now().date()
    matched = []
    for stock in stock_results:
        last_buy = stock.get("last_buy_date") or _last_buy_date(stock)
        if isinstance(last_buy, str):
            last_buy = pd.to_datetime(last_buy).date()
        if last_buy is None:
            continue
        trades = stock.get("trades") or []
        holding = stock.get("is_holding")
        if holding is None:
            holding = bool(trades) and trades[-1].get("type") == "buy"
        recent = (today - last_buy).days <= recent_days
        if holding or recent:
            stock = dict(stock)
            stock["last_buy_date"] = last_buy
            stock["is_holding"] = holding
            matched.append(stock)
    matched.sort(key=lambda x: x.get("last_buy_date") or datetime.min.date(), reverse=True)
    return matched


def print_recent_buy_stocks(recent_buy_stocks, recent_days=RECENT_BUY_DAYS):
    logging.info(f"\n{'=' * 100}")
    logging.info(f"最近有买入的股票（当前持仓，或近{recent_days}天内买入）")
    logging.info(f"{'=' * 100}")
    if not recent_buy_stocks:
        logging.info("没有符合条件的近期买入股票")
        return
    logging.info(f"共 {len(recent_buy_stocks)} 只")
    for i, stock in enumerate(recent_buy_stocks, 1):
        status = "持仓中" if stock.get("is_holding") else "已卖出"
        logging.info(
            f"{i}. {stock.get('stock_code')} {stock.get('stock_name', '')} "
            f"最近买入 {stock.get('last_buy_date')} [{status}] "
            f"总收益 {stock.get('total_return', 0):.2%} 胜率 {stock.get('win_rate', 0):.2%}"
        )


def notify_recent_buy_stocks(recent_buy_stocks, win_rate_th, total_return, recent_days=RECENT_BUY_DAYS):
    mail_lines = [
        f"回测日期: {datetime.now().strftime('%Y-%m-%d')}",
        f"最近30日龙虎榜 | 胜率>={win_rate_th:.0%} 且 总收益>={total_return:.0%}",
        f"再筛：当前持仓或近{recent_days}天内买入",
        f"命中 {len(recent_buy_stocks)} 只",
        "",
        "--------------------------------------",
        " recent buys",
        "--------------------------------------",
    ]
    if not recent_buy_stocks:
        mail_lines.append("(无)")
    else:
        for stock in recent_buy_stocks:
            status = "持仓中" if stock.get("is_holding") else "已卖出"
            mail_lines.append(
                f"{stock.get('stock_code')} {stock.get('stock_name', '')} "
                f"买入 {stock.get('last_buy_date')} [{status}] "
                f"收益 {stock.get('total_return', 0):.2%} 胜率 {stock.get('win_rate', 0):.2%}"
            )
    try:
        efi_email.send(mail_lines)
    except Exception as e:
        logging.error(f"发送近期买入邮件失败: {e}")


def print_high_performance_details(high_performance_stocks,win_rate_th = 0.60, total_return = 2.00):
    """
    打印高性能股票的详细信息
    """
    if not high_performance_stocks:
        logging.info(f"❌ 没有找到胜率超过{win_rate_th}且收益率超过{total_return}%的股票")
        return

    logging.info(f"\n{'=' * 100}")
    logging.info(f"🎯 高性能股票筛选结果（胜率>{win_rate_th} 且 收益率>{total_return}%）")
    logging.info(f"{'=' * 100}")
    logging.info(f"共找到 {len(high_performance_stocks)} 只符合条件的股票")
    logging.info(f"{'=' * 100}")

    # 按收益率排序
    high_performance_stocks.sort(key=lambda x: x['total_return'], reverse=True)

    for i, stock in enumerate(high_performance_stocks, 1):
        logging.info(f"\n📈 第 {i} 只高性能股票:")
        logging.info(f"  股票代码: {stock['stock_code']}")
        logging.info(f"  股票名称: {stock.get('stock_name', 'N/A')}")
        logging.info(f"  胜率: {stock['win_rate']:.2%}")
        logging.info(f"  总收益率: {stock['total_return']:.2%}")
        logging.info(f"  年化收益率: {stock.get('annual_return', 0):.2%}")
        logging.info(f"  夏普比率: {stock.get('sharpe_ratio', 0):.2f}")
        logging.info(f"  交易次数: {stock.get('number_of_trades', 0)}")
        logging.info(f"  初始资金: {stock['initial_capital']:,.2f}")
        logging.info(f"  最终资金: {stock['final_capital']:,.2f}")

        # 如果有持仓信息，显示持仓状态
        if 'position' in stock:
            logging.info(f"  持仓状态: {stock['position']}")

        logging.info("-" * 80)


def generate_performance_report(high_performance_stocks):
    """
    生成高性能股票的统计报告
    """
    if not high_performance_stocks:
        return

    logging.info(f"\n{'=' * 80}")
    logging.info("📊 高性能股票统计报告")
    logging.info(f"{'=' * 80}")

    # 基本统计
    total_stocks = len(high_performance_stocks)
    avg_win_rate = sum(stock['win_rate'] for stock in high_performance_stocks) / total_stocks
    avg_return = sum(stock['total_return'] for stock in high_performance_stocks) / total_stocks
    max_return = max(stock['total_return'] for stock in high_performance_stocks)
    min_return = min(stock['total_return'] for stock in high_performance_stocks)

    logging.info(f"股票数量: {total_stocks}")
    logging.info(f"平均胜率: {avg_win_rate:.2%}")
    logging.info(f"平均收益率: {avg_return:.2%}")
    logging.info(f"最高收益率: {max_return:.2%}")
    logging.info(f"最低收益率: {min_return:.2%}")

    # 收益率分布
    return_ranges = {
        "000%-100%": 0,
        "100%-200%": 0,
        "200%-300%": 0,
        "300%-400%": 0,
        "400%-500%": 0,
        "500%以上": 0
    }

    for stock in high_performance_stocks:
        return_rate = stock['total_return']
        if 0.0 <= return_rate < 1.0:
            return_ranges["000%-100%"] += 1
        elif 1.0 <= return_rate < 2.0:
            return_ranges["100%-200%"] += 1
        elif 2.0 <= return_rate < 3.0:
            return_ranges["200%-300%"] += 1
        elif 3.0 <= return_rate < 4.0:
            return_ranges["300%-400%"] += 1
        elif 4.0 <= return_rate < 5.0:
            return_ranges["400%-500%"] += 1
        else:
            return_ranges["500%以上"] += 1

    logging.info(f"\n收益率分布:")
    for range_name, count in return_ranges.items():
        if count > 0:
            percentage = count / total_stocks * 100
            logging.info(f"  {range_name}: {count}只 ({percentage:.1f}%)")


def save_high_performance_stocks(high_performance_stocks, win_rate_th = 0.60, total_return = 2.00, filename=None):
    """
    将高性能股票保存到文件
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"high_performance_stocks_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"高性能股票列表（胜率> {win_rate_th} 且 收益率 > {total_return}）\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"股票数量: {len(high_performance_stocks)}\n\n")

            for i, stock in enumerate(high_performance_stocks, 1):
                f.write(f"{i}. {stock['stock_code']} - {stock.get('stock_name', 'N/A')}\n")
                f.write(f"   胜率: {stock['win_rate']:.2%}\n")
                f.write(f"   收益率: {stock['total_return']:.2%}\n")
                f.write(f"   年化收益率: {stock.get('annual_return', 0):.2%}\n")
                f.write(f"   夏普比率: {stock.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"   交易次数: {stock.get('number_of_trades', 0)}\n")
                f.write("-" * 40 + "\n")

        logging.info(f"💾 高性能股票列表已保存到: {filename}")

    except Exception as e:
        logging.error(f"保存文件时出错: {str(e)}")
