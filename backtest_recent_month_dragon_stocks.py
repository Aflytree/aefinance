import logging
from datetime import datetime, timedelta
import pandas as pd
import time
import util
import backtest_strategy


def backtest_recent_month_dragon_stocks():
    """
    自动遍历最近一个月每天的龙虎榜股票并进行回测
    """
    # 获取当前日期
    end_date = datetime.now()
    # 计算一个月前的日期
    start_date = end_date - timedelta(days=60)

    logging.info(
        f"开始回测最近一个月龙虎榜股票，时间范围: {start_date.strftime('%Y%m%d')} 到 {end_date.strftime('%Y%m%d')}")

    # 生成最近一个月的所有日期
    date_range = pd.date_range(start=start_date, end=end_date)

    all_stock_results = []  # 改为存储所有单个股票的结果
    summary_stats = []
    total_stocks_tested = 0
    successful_tests = 0
    print(date_range)

    for current_date in date_range:
        date_str = current_date.strftime('%Y%m%d')

        # 跳过周末（可选，因为股票市场周末不交易）
        if current_date.weekday() >= 5:  # 5=周六, 6=周日
            continue

        try:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"处理日期: {date_str}")
            logging.info(f"{'=' * 60}")

            # 获取当天的龙虎榜股票
            stock_codes, day_dragons = util.get_dragon_tiger_stocks(date=date_str)

            if not stock_codes:
                logging.info(f"日期 {date_str} 没有龙虎榜数据或数据获取失败")
                continue

            logging.info(f"日期 {date_str} 龙虎榜股票数量: {len(stock_codes)}")
            logging.info(f"股票代码: {stock_codes}")

            daily_results = []
            daily_trades = []
            daily_last_buys = []
            # import pdb;pdb.set_trace()
            # stock_codes = ['000014', '600617', '600748', '601608', '603011', '605255', '000027', '000592']
            for i, code in enumerate(stock_codes, 1):
                try:
                    logging.info(f"\n[{i}/{len(stock_codes)}] 回测股票: {code}")

                    # 进行回测
                    results = backtest_strategy.backtest_strategy(
                        code,
                        bg='20210323',  # 回测开始日期
                        initial_capital_=1000000,
                        target_return_=0.11,
                        stop_loss_=-0.03,
                        init_stop_n_times=0
                    )

                    if results is None:
                        logging.info(f"股票 {code} 回测无结果，可能数据不足")
                        continue

                    # 打印回测结果
                    # util.print_backtest_results(results)

                    # 收集结果 - 将单个股票结果添加到总列表中
                    all_stock_results.append(results)
                    daily_results.append(results)
                    daily_trades.append(util.trade_daily(code, results))
                    daily_last_buys.append(util.last_busy(code, results))

                    successful_tests += 1
                    logging.info(f"股票 {code} 回测完成")

                    # 添加短暂延迟，避免请求过于频繁
                    time.sleep(0.5)

                except Exception as e:
                    logging.error(f"回测股票 {code} 时出错: {str(e)}")
                    continue

            total_stocks_tested += len(stock_codes)

            # 汇总当天的回测结果
            if daily_results:
                date_summary = {
                    'date': date_str,
                    'stock_count': len(stock_codes),
                    'successful_backtests': len(daily_results),
                    'results': daily_results,
                    'trades': daily_trades,
                    'last_buys': daily_last_buys
                }
                summary_stats.append(date_summary)

                logging.info(f"\n✅ 日期 {date_str} 回测完成: {len(daily_results)}/{len(stock_codes)} 只股票回测成功")
            else:
                logging.info(f"\n❌ 日期 {date_str} 所有股票回测均失败")

        except Exception as e:
            logging.error(f"处理日期 {date_str} 时出错: {str(e)}")
            continue

    # 生成月度汇总报告
    generate_monthly_report(summary_stats, total_stocks_tested, successful_tests, all_stock_results)

    return all_stock_results  # 返回所有单个股票的结果


def generate_monthly_report(summary_stats, total_stocks, successful_tests, all_stock_results):
    """
    生成月度回测汇总报告
    """
    logging.info(f"\n{'=' * 80}")
    logging.info("📊 最近一个月龙虎榜股票回测汇总报告")
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
        high_performance_stocks = filter_high_performance_stocks(all_stock_results)
        print_high_performance_details(high_performance_stocks)
        generate_performance_report(high_performance_stocks)
        save_high_performance_stocks(high_performance_stocks)
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
        results = backtest_recent_month_dragon_stocks()

        logging.info("\n🎉 最近一个月龙虎榜股票回测完成！")
        return results

    except Exception as e:
        logging.error(f"回测执行过程中发生错误: {str(e)}")
        return None


def filter_high_performance_stocks(all_stock_results):
    """
    筛选胜率超过70%且收益率超过200%的高性能股票
    """
    high_performance_stocks = []

    for stock_result in all_stock_results:
        try:
            # 检查必要字段是否存在
            if not all(key in stock_result for key in ['win_rate', 'total_return', 'stock_code']):
                continue

            win_rate = stock_result['win_rate']
            total_return = stock_result['total_return']

            # 筛选条件：胜率 > 70% 且 总收益率 > 200%
            if win_rate > 0.50 and total_return > 2.00:
                high_performance_stocks.append(stock_result)

        except (KeyError, TypeError, ValueError) as e:
            logging.warning(f"处理股票 {stock_result.get('stock_code', '未知')} 数据时出错: {e}")
            continue

    return high_performance_stocks


def print_high_performance_details(high_performance_stocks):
    """
    打印高性能股票的详细信息
    """
    if not high_performance_stocks:
        logging.info("❌ 没有找到胜率超过70%且收益率超过200%的股票")
        return

    logging.info(f"\n{'=' * 100}")
    logging.info("🎯 高性能股票筛选结果（胜率>70% 且 收益率>200%）")
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
        "200%-300%": 0,
        "300%-400%": 0,
        "400%-500%": 0,
        "500%以上": 0
    }

    for stock in high_performance_stocks:
        return_rate = stock['total_return']
        if 2.0 <= return_rate < 3.0:
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


def save_high_performance_stocks(high_performance_stocks, filename=None):
    """
    将高性能股票保存到文件
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"high_performance_stocks_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("高性能股票列表（胜率>70% 且 收益率>200%）\n")
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

