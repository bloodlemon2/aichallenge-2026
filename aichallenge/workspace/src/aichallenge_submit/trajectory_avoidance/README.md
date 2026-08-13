# trajectory_avoidance

`/aichallenge/objects` の `x, y, z, radius` 配列から、前フレームとの差分で他車の速度を推定します。前方車両を等速運動として予測し、基準レーシングラインとの衝突見込みがある場合だけ、左・右へ滑らかにオフセットした候補のうち安全余裕が大きい軌道を出力します。

起動時は次のように接続されます。

```
simple_trajectory_generator
  -> /planning/scenario_planning/base_trajectory
  -> trajectory_avoidance
  -> /planning/scenario_planning/trajectory
  -> Pure Pursuit / MPC
```

主なパラメータは `detection_distance`（20 m）、`collision_horizon`（4 s）、`safety_margin`（0.45 m）、`avoidance_offset`（1.0 m）です。コース幅・車両サイズに合わせて後ろの二つを調整してください。
