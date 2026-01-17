import numpy as np
import pandas as pd
from flight_environment import FlightEnvironment
from path_planner import AStarPlanner, RRTPlanner
from trajectory_generator import (
    TrajectoryGenerator,
    QuinticPolynomialTrajectory,
    evaluate_trajectory_quality,
)
import time


def run_experiment(planner_name, planner_class, planner_kwargs, 
                   traj_name, traj_class, 
                   start, goal, num_runs=5):
    all_metrics = []
    
    for run in range(num_runs):
        print(f"\n[{planner_name} + {traj_name}] Run {run + 1}/{num_runs}")
        
        try:
            env = FlightEnvironment(140, start, goal)
            planner = planner_class(env, **planner_kwargs)
            plan_start = time.time()
            raw_path = planner.plan(start, goal)
            path = planner.smooth_path(raw_path, iterations=50)
            plan_time = time.time() - plan_start
            
            path_metrics = planner.evaluate_path_quality(path)

            traj_gen = traj_class(path, velocity=2.0)

            traj_start = time.time()
            t_array, trajectory, waypoint_times = traj_gen.generate(num_points=200)
            traj_time = time.time() - traj_start
            traj_metrics = evaluate_trajectory_quality(trajectory, t_array, path)
            
            combined_metrics = {
                # 路径指标
                'total_length': path_metrics['total_length'],
                'num_waypoints': path_metrics['num_waypoints'],
                'avg_segment_length': path_metrics['avg_segment_length'],
                'avg_turn_angle': path_metrics['avg_turn_angle'],
                'max_turn_angle': path_metrics['max_turn_angle'],
                
                # 轨迹指标
                'flight_time': t_array[-1],
                'max_velocity': traj_metrics['max_velocity'],
                'avg_velocity': traj_metrics['avg_velocity'],
                'max_acceleration': traj_metrics['max_acceleration'],
                'max_jerk': traj_metrics['max_jerk'],
                'jerk_cost': traj_metrics['jerk_cost'],
                'max_waypoint_deviation': traj_metrics['max_waypoint_deviation'],
                
                # 计算时间
                'planning_time': plan_time,
                'trajectory_time': traj_time,
            }
            
            all_metrics.append(combined_metrics)
            
            print(f"  ✓ Path: {path_metrics['total_length']:.2f}m, "
                  f"Waypoints: {path_metrics['num_waypoints']}, "
                  f"Time: {t_array[-1]:.2f}s, "
                  f"Jerk: {traj_metrics['jerk_cost']:.2f}")
            
        except Exception as e:
            print(f"  ✗ Run {run + 1} failed: {e}")
            continue
    
    if not all_metrics:
        print(f"  ERROR: All runs failed for {planner_name} + {traj_name}")
        return None
    
    stats_metrics = {}
    metric_keys = all_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        stats_metrics[f'{key}_mean'] = np.mean(values)
        stats_metrics[f'{key}_std'] = np.std(values)
    
    stats_metrics['successful_runs'] = len(all_metrics)
    
    return stats_metrics


def main():
    start = (1, 2, 0)
    goal = (18, 18, 3)
    num_runs = 10
    
    experiments = [
        # A* + Cubic Spline
        {
            'planner_name': 'A*',
            'planner_class': AStarPlanner,
            'planner_kwargs': {'resolution': 0.5},
            'traj_name': 'Cubic Spline',
            'traj_class': TrajectoryGenerator
        },
        # A* + Quintic Polynomial
        {
            'planner_name': 'A*',
            'planner_class': AStarPlanner,
            'planner_kwargs': {'resolution': 0.5},
            'traj_name': 'Quintic Polynomial',
            'traj_class': QuinticPolynomialTrajectory
        },
        # RRT + Cubic Spline
        {
            'planner_name': 'RRT',
            'planner_class': RRTPlanner,
            'planner_kwargs': {'step_size': 0.5},
            'traj_name': 'Cubic Spline',
            'traj_class': TrajectoryGenerator
        },
        # RRT + Quintic Polynomial
        {
            'planner_name': 'RRT',
            'planner_class': RRTPlanner,
            'planner_kwargs': {'step_size': 0.5},
            'traj_name': 'Quintic Polynomial',
            'traj_class': QuinticPolynomialTrajectory
        }
    ]
    
    results_dict = {}
    
    print("="*80)
    print(f"Starting Benchmark: {len(experiments)} Configurations × {num_runs} Runs")
    print("="*80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(experiments)}: {exp['planner_name']} + {exp['traj_name']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        stats_metrics = run_experiment(
            planner_name=exp['planner_name'],
            planner_class=exp['planner_class'],
            planner_kwargs=exp['planner_kwargs'],
            traj_name=exp['traj_name'],
            traj_class=exp['traj_class'],
            start=start,
            goal=goal,
            num_runs=num_runs
        )
        
        elapsed_time = time.time() - start_time
        
        if stats_metrics is not None:
            config_name = f"{exp['planner_name']} + {exp['traj_name']}"
            
            stats_metrics['planner'] = exp['planner_name']
            stats_metrics['trajectory_generator'] = exp['traj_name']
            stats_metrics['experiment_time'] = elapsed_time
            
            results_dict[config_name] = stats_metrics
            
            print(f"\n✓ Completed in {elapsed_time:.1f} seconds")
        else:
            print(f"\n✗ Experiment failed")
    
    if not results_dict:
        print("\n❌ All experiments failed!")
        return
    
    df = pd.DataFrame(results_dict)
    
    metric_order = [
        'planner',
        'trajectory_generator',
        'successful_runs',
        'experiment_time',
        'planning_time_mean',
        'planning_time_std',
        'trajectory_time_mean',
        'trajectory_time_std',
        'total_length_mean',
        'total_length_std',
        'num_waypoints_mean',
        'num_waypoints_std',
        'avg_segment_length_mean',
        'avg_segment_length_std',
        'avg_turn_angle_mean',
        'avg_turn_angle_std',
        'max_turn_angle_mean',
        'max_turn_angle_std',
        'flight_time_mean',
        'flight_time_std',
        'max_velocity_mean',
        'max_velocity_std',
        'avg_velocity_mean',
        'avg_velocity_std',
        'max_acceleration_mean',
        'max_acceleration_std',
        'max_jerk_mean',
        'max_jerk_std',
        'jerk_cost_mean',
        'jerk_cost_std',
        'max_waypoint_deviation_mean',
        'max_waypoint_deviation_std',
    ]
    
    # 只保留存在的行
    metric_order = [m for m in metric_order if m in df.index]
    df = df.reindex(metric_order)
    
    output_file = 'benchmark_results.csv'
    df.to_csv(output_file, float_format='%.4f')
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Successful experiments: {len(results_dict)}")
    print(f"Failed experiments: {len(experiments) - len(results_dict)}")
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string())
    
    # 对比分析
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    df_for_analysis = df.T  # 转置用于分组分析
    
    # 按路径规划器分组
    print("\n--- By Path Planner ---")
    planner_groups = df_for_analysis.groupby('planner').agg({
        'total_length_mean': 'mean',
        'num_waypoints_mean': 'mean',
        'flight_time_mean': 'mean',
        'jerk_cost_mean': 'mean',
        'jerk_cost_std': 'mean',  # 平均标准差
    })
    print(planner_groups.to_string())
    
    print("\n--- By Trajectory Generator ---")
    traj_groups = df_for_analysis.groupby('trajectory_generator').agg({
        'flight_time_mean': 'mean',
        'max_velocity_mean': 'mean',
        'max_acceleration_mean': 'mean',
        'max_jerk_mean': 'mean',
        'max_jerk_std': 'mean',
        'jerk_cost_mean': 'mean',
        'jerk_cost_std': 'mean',
    })
    print(traj_groups.to_string())
    
    print("\n--- Best Configurations ---")
    
    idx_shortest = df_for_analysis['total_length_mean'].idxmin()
    print(f"Shortest path:        {idx_shortest} "
          f"({df_for_analysis.loc[idx_shortest, 'total_length_mean']:.2f} ± "
          f"{df_for_analysis.loc[idx_shortest, 'total_length_std']:.2f} m)")
    
    idx_fastest = df_for_analysis['flight_time_mean'].idxmin()
    print(f"Fastest flight:       {idx_fastest} "
          f"({df_for_analysis.loc[idx_fastest, 'flight_time_mean']:.2f} ± "
          f"{df_for_analysis.loc[idx_fastest, 'flight_time_std']:.2f} s)")
    
    idx_smoothest = df_for_analysis['jerk_cost_mean'].idxmin()
    print(f"Smoothest trajectory: {idx_smoothest} "
          f"({df_for_analysis.loc[idx_smoothest, 'jerk_cost_mean']:.2f} ± "
          f"{df_for_analysis.loc[idx_smoothest, 'jerk_cost_std']:.2f})")
    
    print("\n--- Most Stable Configurations ---")
    
    idx_stable_jerk = df_for_analysis['jerk_cost_std'].idxmin()
    print(f"Most stable jerk:     {idx_stable_jerk} "
          f"(std = {df_for_analysis.loc[idx_stable_jerk, 'jerk_cost_std']:.2f})")
    
    idx_stable_time = df_for_analysis['flight_time_std'].idxmin()
    print(f"Most stable time:     {idx_stable_time} "
          f"(std = {df_for_analysis.loc[idx_stable_time, 'flight_time_std']:.2f} s)")
    
    print("\n" + "="*80)
    print("Cubic Spline vs Quintic Polynomial Comparison")
    print("="*80)
    
    cubic_configs = df_for_analysis[df_for_analysis['trajectory_generator'] == 'Cubic Spline']
    quintic_configs = df_for_analysis[df_for_analysis['trajectory_generator'] == 'Quintic Polynomial']
    
    if len(cubic_configs) > 0 and len(quintic_configs) > 0:
        print("\nAverage improvements (Quintic vs Cubic):")
        
        metrics_to_compare = [
            ('max_jerk_mean', 'Max Jerk'),
            ('jerk_cost_mean', 'Jerk Cost'),
            ('max_acceleration_mean', 'Max Acceleration'),
            ('max_waypoint_deviation_mean', 'Max Waypoint Deviation')
        ]
        
        for metric, label in metrics_to_compare:
            cubic_mean = cubic_configs[metric].mean()
            quintic_mean = quintic_configs[metric].mean()
            improvement = (cubic_mean - quintic_mean) / cubic_mean * 100
            
            metric_std = metric.replace('_mean', '_std')
            cubic_std = cubic_configs[metric_std].mean()
            quintic_std = quintic_configs[metric_std].mean()
            
            print(f"\n  {label}:")
            print(f"    Cubic:   {cubic_mean:.4f} ± {cubic_std:.4f}")
            print(f"    Quintic: {quintic_mean:.4f} ± {quintic_std:.4f}")
            print(f"    Improvement: {improvement:+6.2f}%")
            
            if quintic_std < cubic_std:
                stability_improvement = (cubic_std - quintic_std) / cubic_std * 100
                print(f"    Stability: {stability_improvement:+6.2f}% more stable")
            else:
                stability_decline = (quintic_std - cubic_std) / cubic_std * 100
                print(f"    Stability: {stability_decline:+6.2f}% less stable")


if __name__ == "__main__":
    main()
