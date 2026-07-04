from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


ACTION_NAMES = (
    "balanced",
    "high_press",
    "low_block",
    "possession",
    "direct_attack",
    "counter",
    "conserve",
    "aggressive",
)

GOAL_HALF_WIDTH = 0.11
MAX_STOPPAGE_MINUTES = 8


@dataclass
class TeamStats:
    goals: int = 0
    xg: float = 0.0
    shots: int = 0
    shots_on_target: int = 0
    passes: int = 0
    completed_passes: int = 0
    fouls: int = 0
    offsides: int = 0
    yellows: int = 0
    reds: int = 0
    corners: int = 0
    free_kicks: int = 0
    penalties: int = 0
    throw_ins: int = 0
    goal_kicks: int = 0
    saves: int = 0
    tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    substitutions: int = 0
    injuries: int = 0
    possession_ticks: int = 0
    stamina: float = 1.0


@dataclass
class SoccerState:
    minute: int = 0
    ball_x: float = 0.0
    ball_y: float = 0.0
    possession: int = 0
    phase: str = "kickoff"
    blue: TeamStats = field(default_factory=TeamStats)
    red: TeamStats = field(default_factory=TeamStats)
    last_event: str = "kickoff"
    stopped_time: float = 0.0


class TacticalSoccerEnv:
    """Small Gym-like tactical soccer environment for fast RL episodes.

    The browser simulator handles detailed animation. This environment is the
    training backend: each step is roughly one compressed minute with team-level
    tactical actions and IFAB-inspired rule events.
    """

    def __init__(self, *, seed: int | None = None, max_minutes: int = 90) -> None:
        self.random = random.Random(seed)
        self.max_minutes = int(max_minutes)
        self.state = SoccerState()
        self.replay: list[dict] = []
        self.last_reward_terms: dict[str, float] = {}

    def reset(self) -> list[float]:
        self.state = SoccerState(
            possession=1 if self.random.random() < 0.5 else -1,
            phase="kickoff",
            last_event="kickoff",
        )
        self.replay = []
        self._record_frame()
        return self.observation()

    def observation(self) -> list[float]:
        s = self.state
        goal_diff = s.blue.goals - s.red.goals
        shot_diff = s.blue.shots - s.red.shots
        foul_diff = s.blue.fouls - s.red.fouls
        save_diff = s.blue.saves - s.red.saves
        card_diff = (s.red.yellows + s.red.reds * 2) - (s.blue.yellows + s.blue.reds * 2)
        restart_pressure = self._restart_pressure()
        return [
            s.minute / max(1, self.max_minutes + MAX_STOPPAGE_MINUTES),
            s.ball_x,
            s.ball_y,
            float(s.possession),
            max(-3.0, min(3.0, goal_diff)) / 3.0,
            max(-10.0, min(10.0, shot_diff)) / 10.0,
            s.blue.stamina,
            s.red.stamina,
            s.blue.xg - s.red.xg,
            max(-8.0, min(8.0, foul_diff)) / 8.0,
            s.blue.offsides / 8.0,
            1.0,
            (11 - s.blue.reds) / 11.0,
            (11 - s.red.reds) / 11.0,
            restart_pressure,
            min(MAX_STOPPAGE_MINUTES, math.ceil(s.stopped_time)) / MAX_STOPPAGE_MINUTES,
            max(-8.0, min(8.0, save_diff)) / 8.0,
            s.ball_x,
            max(-8.0, min(8.0, card_diff)) / 8.0,
        ]

    def step(self, blue_action: int, red_action: int) -> tuple[list[float], float, bool, dict]:
        s = self.state
        before = self._metric_snapshot()
        blue_action = int(max(0, min(len(ACTION_NAMES) - 1, blue_action)))
        red_action = int(max(0, min(len(ACTION_NAMES) - 1, red_action)))

        self._tick_possession()
        self._apply_stamina("blue", blue_action)
        self._apply_stamina("red", red_action)
        self._maybe_substitute("blue")
        self._maybe_substitute("red")
        self._resolve_minute(blue_action, red_action)

        s.minute += 1
        if s.minute == 45:
            s.phase = "half-time kickoff"
            s.possession *= -1
            s.ball_x = 0.0
            s.ball_y = 0.0
            s.last_event = "half-time kickoff"
            s.stopped_time += 0.2
        match_limit = self.max_minutes + min(MAX_STOPPAGE_MINUTES, math.ceil(s.stopped_time))
        done = s.minute >= match_limit or self._too_few_players()

        terms = self._reward_terms(before, done=done)
        reward = sum(terms.values())
        self.last_reward_terms = terms

        info = self.info()
        info["blue_action"] = ACTION_NAMES[blue_action]
        info["red_action"] = ACTION_NAMES[red_action]
        info["reward_terms"] = terms
        self._record_frame(blue_action=blue_action, red_action=red_action, reward=reward, reward_terms=terms)
        return self.observation(), reward, done, info

    def info(self) -> dict:
        s = self.state
        return {
            "minute": s.minute,
            "phase": s.phase,
            "last_event": s.last_event,
            "score": {"blue": s.blue.goals, "red": s.red.goals},
            "xg": {"blue": round(s.blue.xg, 3), "red": round(s.red.xg, 3)},
            "shots": {"blue": s.blue.shots, "red": s.red.shots},
            "shots_on_target": {"blue": s.blue.shots_on_target, "red": s.red.shots_on_target},
            "pass_completion": {
                "blue": self._rate(s.blue.completed_passes, s.blue.passes),
                "red": self._rate(s.red.completed_passes, s.red.passes),
            },
            "discipline": {
                "blue_fouls": s.blue.fouls,
                "red_fouls": s.red.fouls,
                "blue_yellows": s.blue.yellows,
                "red_yellows": s.red.yellows,
                "blue_reds": s.blue.reds,
                "red_reds": s.red.reds,
                "blue_offsides": s.blue.offsides,
                "red_offsides": s.red.offsides,
            },
            "stamina": {"blue": round(s.blue.stamina, 3), "red": round(s.red.stamina, 3)},
            "possession": {"blue": self._rate(s.blue.possession_ticks, max(1, s.minute)), "red": self._rate(s.red.possession_ticks, max(1, s.minute))},
            "restarts": {
                "blue_corners": s.blue.corners,
                "red_corners": s.red.corners,
                "blue_free_kicks": s.blue.free_kicks,
                "red_free_kicks": s.red.free_kicks,
                "blue_penalties": s.blue.penalties,
                "red_penalties": s.red.penalties,
                "blue_throw_ins": s.blue.throw_ins,
                "red_throw_ins": s.red.throw_ins,
                "blue_goal_kicks": s.blue.goal_kicks,
                "red_goal_kicks": s.red.goal_kicks,
            },
            "match": {
                "active_players": {"blue": 11 - s.blue.reds, "red": 11 - s.red.reds},
                "added_time": min(MAX_STOPPAGE_MINUTES, math.ceil(s.stopped_time)),
                "stopped_time": round(s.stopped_time, 2),
                "substitutions": {"blue": s.blue.substitutions, "red": s.red.substitutions},
                "injuries": {"blue": s.blue.injuries, "red": s.red.injuries},
            },
            "goalkeeping": {
                "blue_saves": s.blue.saves,
                "red_saves": s.red.saves,
            },
            "defending": {
                "blue_tackles": s.blue.tackles,
                "red_tackles": s.red.tackles,
                "blue_interceptions": s.blue.interceptions,
                "red_interceptions": s.red.interceptions,
                "blue_clearances": s.blue.clearances,
                "red_clearances": s.red.clearances,
            },
            "reward_terms": dict(self.last_reward_terms),
        }

    def _metric_snapshot(self) -> dict:
        s = self.state
        return {
            "goal_diff": s.blue.goals - s.red.goals,
            "xg_diff": s.blue.xg - s.red.xg,
            "shot_diff": s.blue.shots - s.red.shots,
            "sot_diff": s.blue.shots_on_target - s.red.shots_on_target,
            "def_actions": s.blue.tackles + s.blue.interceptions + s.blue.clearances,
            "blue_fouls": s.blue.fouls,
            "blue_offsides": s.blue.offsides,
            "blue_yellows": s.blue.yellows,
            "blue_reds": s.blue.reds,
            "blue_injuries": s.blue.injuries,
            "blue_stamina": s.blue.stamina,
            "red_stamina": s.red.stamina,
        }

    def _reward_terms(self, before: dict, *, done: bool) -> dict[str, float]:
        s = self.state
        goal_diff = s.blue.goals - s.red.goals
        xg_diff = s.blue.xg - s.red.xg
        shot_diff = s.blue.shots - s.red.shots
        sot_diff = s.blue.shots_on_target - s.red.shots_on_target
        def_actions = s.blue.tackles + s.blue.interceptions + s.blue.clearances
        discipline_delta = (
            (s.blue.fouls - before["blue_fouls"]) * 0.018
            + (s.blue.offsides - before["blue_offsides"]) * 0.022
            + (s.blue.yellows - before["blue_yellows"]) * 0.12
            + (s.blue.reds - before["blue_reds"]) * 0.55
            + (s.blue.injuries - before["blue_injuries"]) * 0.08
        )
        terms = {
            "score": (goal_diff - before["goal_diff"]) * 6.0,
            "xg": (xg_diff - before["xg_diff"]) * 1.15,
            "shot_quality": (sot_diff - before["sot_diff"]) * 0.1 + (shot_diff - before["shot_diff"]) * 0.03,
            "territory": max(-1.0, min(1.0, s.ball_x)) * 0.018,
            "possession": 0.012 if s.possession == 1 else -0.012,
            "defense": (def_actions - before["def_actions"]) * 0.018,
            "discipline": -discipline_delta,
            "stamina": ((s.blue.stamina - s.red.stamina) - (before["blue_stamina"] - before["red_stamina"])) * 0.08,
            "result": goal_diff * 2.0 if done else 0.0,
        }
        return {key: round(value, 4) for key, value in terms.items()}

    def _resolve_minute(self, blue_action: int, red_action: int) -> None:
        s = self.state
        attacking = "blue" if s.possession >= 0 else "red"
        defending = "red" if attacking == "blue" else "blue"
        attack_action = blue_action if attacking == "blue" else red_action
        defend_action = red_action if attacking == "blue" else blue_action
        attack = s.blue if attacking == "blue" else s.red
        defend = s.red if attacking == "blue" else s.blue
        sign = 1.0 if attacking == "blue" else -1.0
        phase = s.phase
        restart = self._restart_profile(phase, attack_action)

        pass_risk = self._pass_risk(attack_action)
        pass_quality = self._pass_quality(attack_action, attack.stamina) + restart["pass_bonus"]
        press = self._press_intensity(defend_action, defend.stamina)
        foul_risk = self._foul_risk(defend_action, defend.stamina) * restart["foul_mult"]

        attack.passes += 1
        completed = self.random.random() < max(0.28, min(0.92, pass_quality - press * 0.12))
        if completed:
            attack.completed_passes += 1
            progress = self._progress(attack_action, defend_action) * attack.stamina * restart["progress_mult"]
            new_x = s.ball_x + sign * progress
            new_y = s.ball_y + self.random.uniform(-0.18, 0.18) * restart["width_mult"]
            if self._offside_offence(phase, attacking, attack_action, new_x, pass_risk):
                attack.offsides += 1
                self._restart(defending, "offside free kick", x=new_x, y=new_y)
                return
            if self._ball_out(attacking, defending, new_x, new_y, last_touch=attacking):
                return
            s.ball_x = max(-1.0, min(1.0, new_x))
            s.ball_y = max(-1.0, min(1.0, new_y))
            s.phase = "open"
            s.last_event = f"{attacking} pass"
        else:
            if self._offside_offence(phase, attacking, attack_action, s.ball_x + sign * 0.08, pass_risk):
                attack.offsides += 1
                self._restart(defending, "offside free kick")
            else:
                s.possession *= -1
                s.phase = "open"
                defend.interceptions += 1
                s.last_event = f"{defending} interception"
            return

        if self.random.random() < foul_risk:
            defend.fouls += 1
            s.stopped_time += 0.15
            caution_risk = self._caution_risk(defend_action, defend.stamina, s.ball_x * sign)
            if self.random.random() < caution_risk:
                defend.yellows += 1
                s.stopped_time += 0.25
            if self.random.random() < max(0.008, caution_risk * 0.12):
                defend.reds += 1
                s.stopped_time += 0.4
            if self.random.random() < 0.03 + max(0.0, 0.55 - defend.stamina) * 0.08:
                defend.injuries += 1
                s.stopped_time += 0.45
            if self._in_penalty_area(attacking):
                attack.penalties += 1
                self._restart(attacking, "penalty", x=0.82 * sign, y=0.0)
                self._shot(attacking, penalty=True)
            elif self._advantage(attacking, attack_action):
                s.last_event = f"{attacking} advantage"
            else:
                attack.free_kicks += 1
                restart_phase = "indirect free kick" if self.random.random() < 0.18 else "direct free kick"
                self._restart(attacking, restart_phase)
            return

        set_piece_shot = restart["shot_mult"] > 1.0 and self.random.random() < 0.16 * restart["shot_mult"]
        if set_piece_shot or self.random.random() < self._shot_chance(attack_action, s.ball_x * sign, attack.stamina) * restart["shot_mult"]:
            self._shot(attacking, direct_allowed=not restart["indirect_only"])
            return

        if self.random.random() < max(0.02, press * 0.12 - attack.stamina * 0.04):
            s.possession *= -1
            defend.tackles += 1
            if abs(s.ball_x) > 0.65 and self.random.random() < 0.16:
                defend.clearances += 1
                self._clearance(defending)
                return
            s.last_event = f"{defending} tackle"

    def _shot(self, attacking: str, *, penalty: bool = False, direct_allowed: bool = True) -> None:
        s = self.state
        team = s.blue if attacking == "blue" else s.red
        opponent = s.red if attacking == "blue" else s.blue
        defending = "red" if attacking == "blue" else "blue"
        sign = 1.0 if attacking == "blue" else -1.0
        team.shots += 1
        distance_factor = max(0.05, min(1.0, (s.ball_x * sign + 1.0) / 2.0))
        xg = 0.76 if penalty else max(0.02, min(0.42, 0.04 + distance_factor * 0.25 + team.stamina * 0.08))
        if not direct_allowed:
            xg *= 0.2
        team.xg += xg
        on_target = self.random.random() < min(0.88, xg * 1.9 + 0.18)
        if on_target:
            team.shots_on_target += 1
        if direct_allowed and self.random.random() < xg:
            team.goals += 1
            s.possession *= -1
            s.ball_x = 0.0
            s.ball_y = 0.0
            s.phase = "kickoff"
            s.last_event = f"{attacking} goal"
            s.stopped_time += 0.8
            return
        if on_target:
            opponent.saves += 1
            if self.random.random() < 0.36:
                team.corners += 1
                self._restart(attacking, "corner")
            else:
                self._restart(defending, "goal kick")
            return
        if self.random.random() < 0.14:
            team.corners += 1
            self._restart(attacking, "corner")
        else:
            self._restart(defending, "goal kick")

    def _restart(self, team: str, phase: str, *, x: float | None = None, y: float | None = None) -> None:
        self.state.phase = phase
        self.state.possession = 1 if team == "blue" else -1
        sign = 1.0 if team == "blue" else -1.0
        if phase in ("kickoff", "half-time kickoff"):
            self.state.ball_x = 0.0
            self.state.ball_y = 0.0
        elif phase == "goal kick":
            stats = self.state.blue if team == "blue" else self.state.red
            stats.goal_kicks += 1
            self.state.ball_x = -0.9 * sign
            self.state.ball_y = self.random.uniform(-0.12, 0.12)
            self.state.stopped_time += 0.18
        elif phase == "corner":
            self.state.ball_x = 0.98 * sign
            self.state.ball_y = 1.0 if self.random.random() < 0.5 else -1.0
            self.state.stopped_time += 0.22
        elif phase == "throw-in":
            stats = self.state.blue if team == "blue" else self.state.red
            stats.throw_ins += 1
            self.state.ball_x = max(-0.98, min(0.98, x if x is not None else self.state.ball_x))
            self.state.ball_y = 1.0 if (y or self.state.ball_y) >= 0 else -1.0
            self.state.stopped_time += 0.12
        elif phase in ("direct free kick", "indirect free kick", "offside free kick"):
            self.state.ball_x = max(-0.96, min(0.96, x if x is not None else self.state.ball_x))
            self.state.ball_y = max(-0.86, min(0.86, y if y is not None else self.state.ball_y))
            self.state.stopped_time += 0.2
        elif phase == "penalty":
            self.state.ball_x = 0.82 * sign
            self.state.ball_y = 0.0
            self.state.stopped_time += 0.35
        elif phase == "dropped ball":
            self.state.ball_x = max(-0.8, min(0.8, x if x is not None else self.state.ball_x))
            self.state.ball_y = max(-0.8, min(0.8, y if y is not None else self.state.ball_y))
            self.state.stopped_time += 0.1
        self.state.last_event = f"{team} {phase}"

    def _apply_stamina(self, team: str, action: int) -> None:
        stats = self.state.blue if team == "blue" else self.state.red
        drain = {
            0: 0.0022,
            1: 0.0046,
            2: 0.0016,
            3: 0.0024,
            4: 0.0035,
            5: 0.0031,
            6: -0.0028,
            7: 0.0042,
        }.get(action, 0.002)
        stats.stamina = max(0.22, min(1.0, stats.stamina - drain))

    def _tick_possession(self) -> None:
        if self.state.possession >= 0:
            self.state.blue.possession_ticks += 1
        else:
            self.state.red.possession_ticks += 1

    def _record_frame(self, **extra) -> None:
        frame = self.info()
        frame.update(extra)
        frame["ball"] = {"x": round(self.state.ball_x, 3), "y": round(self.state.ball_y, 3)}
        frame["possession_side"] = "blue" if self.state.possession >= 0 else "red"
        self.replay.append(frame)
        if len(self.replay) > 240:
            self.replay = self.replay[-240:]

    def _too_few_players(self) -> bool:
        return (11 - self.state.blue.reds) < 7 or (11 - self.state.red.reds) < 7

    def _in_penalty_area(self, attacking: str) -> bool:
        return self.state.ball_x > 0.76 if attacking == "blue" else self.state.ball_x < -0.76

    def _advantage(self, attacking: str, action: int) -> bool:
        sign = 1.0 if attacking == "blue" else -1.0
        return action in (4, 5) and self.state.ball_x * sign > 0.35 and self.random.random() < 0.58

    def _maybe_substitute(self, team: str) -> None:
        stats = self.state.blue if team == "blue" else self.state.red
        if self.state.minute < 58 or stats.substitutions >= 5 or stats.stamina > 0.46:
            return
        if self.random.random() < 0.18 + max(0.0, 0.38 - stats.stamina):
            stats.substitutions += 1
            stats.stamina = min(1.0, stats.stamina + self.random.uniform(0.07, 0.13))
            self.state.stopped_time += 0.28
            self.state.last_event = f"{team} substitution"

    def _restart_profile(self, phase: str, action: int) -> dict:
        profile = {
            "pass_bonus": 0.0,
            "progress_mult": 1.0,
            "width_mult": 1.0,
            "shot_mult": 1.0,
            "foul_mult": 1.0,
            "indirect_only": False,
        }
        if phase in ("kickoff", "half-time kickoff"):
            profile.update(pass_bonus=0.12, progress_mult=0.45, shot_mult=0.12)
        elif phase == "goal kick":
            profile.update(pass_bonus=-0.02, progress_mult=1.25, width_mult=0.7, shot_mult=0.08)
            if action in (4, 5):
                profile["pass_bonus"] -= 0.08
                profile["progress_mult"] = 1.65
        elif phase == "throw-in":
            profile.update(pass_bonus=-0.08, progress_mult=0.55, width_mult=0.45, shot_mult=0.18)
        elif phase == "corner":
            profile.update(pass_bonus=-0.16, progress_mult=0.2, width_mult=0.35, shot_mult=2.8, foul_mult=1.18)
        elif phase == "direct free kick":
            danger = max(0.0, abs(self.state.ball_x) - 0.35)
            profile.update(pass_bonus=0.03, progress_mult=0.82, width_mult=0.55, shot_mult=1.35 + danger)
        elif phase == "indirect free kick":
            profile.update(pass_bonus=0.05, progress_mult=0.78, width_mult=0.55, shot_mult=1.15, indirect_only=True)
        elif phase == "offside free kick":
            profile.update(pass_bonus=0.06, progress_mult=0.9, width_mult=0.65, shot_mult=0.4, indirect_only=True)
        elif phase == "dropped ball":
            profile.update(pass_bonus=0.08, progress_mult=0.55, shot_mult=0.1)
        return profile

    def _restart_pressure(self) -> float:
        sign = 1.0 if self.state.possession >= 0 else -1.0
        danger = max(0.0, self.state.ball_x * sign)
        if self.state.phase == "corner":
            return 1.0 * sign
        if self.state.phase == "penalty":
            return 0.95 * sign
        if self.state.phase == "direct free kick":
            return min(0.9, 0.35 + danger * 0.55) * sign
        if self.state.phase in ("indirect free kick", "throw-in"):
            return min(0.55, 0.15 + danger * 0.35) * sign
        if self.state.phase == "goal kick":
            return -0.25 * sign
        return 0.0

    def _offside_offence(self, phase: str, attacking: str, action: int, target_x: float, pass_risk: float) -> bool:
        if phase in ("goal kick", "throw-in", "corner"):
            return False
        sign = 1.0 if attacking == "blue" else -1.0
        if target_x * sign <= 0.34 or action not in (4, 5, 7):
            return False
        line_depth = 0.44 + self._defensive_line_depth(action) * 0.24
        if target_x * sign < line_depth:
            return False
        return self.random.random() < min(0.34, 0.08 + pass_risk * 0.22 + (target_x * sign - line_depth) * 0.18)

    def _ball_out(self, attacking: str, defending: str, x: float, y: float, *, last_touch: str) -> bool:
        if abs(y) > 1.0:
            restart_team = defending if last_touch == attacking else attacking
            self._restart(restart_team, "throw-in", x=x, y=y)
            return True
        sign = 1.0 if attacking == "blue" else -1.0
        if x * sign > 1.0:
            if abs(y) <= GOAL_HALF_WIDTH and last_touch != attacking:
                team = self.state.blue if attacking == "blue" else self.state.red
                team.goals += 1
                self._restart(defending, "kickoff")
                self.state.last_event = f"{attacking} own-pressure goal"
                self.state.stopped_time += 0.8
            elif last_touch == attacking:
                self._restart(defending, "goal kick", x=x, y=y)
            else:
                team = self.state.blue if attacking == "blue" else self.state.red
                team.corners += 1
                self._restart(attacking, "corner", x=x, y=y)
            return True
        if x * sign < -1.0:
            if last_touch == attacking:
                other = self.state.red if attacking == "blue" else self.state.blue
                other.corners += 1
                self._restart(defending, "corner", x=x, y=y)
            else:
                self._restart(attacking, "goal kick", x=x, y=y)
            return True
        return False

    def _clearance(self, defending: str) -> None:
        s = self.state
        attacking = "red" if defending == "blue" else "blue"
        sign = 1.0 if defending == "blue" else -1.0
        x = s.ball_x + sign * self.random.uniform(0.28, 0.52)
        y = s.ball_y + self.random.uniform(-0.45, 0.45)
        if self._ball_out(attacking, defending, x, y, last_touch=defending):
            return
        s.ball_x = max(-1.0, min(1.0, x))
        s.ball_y = max(-1.0, min(1.0, y))
        s.last_event = f"{defending} clearance"

    @staticmethod
    def _caution_risk(action: int, stamina: float, signed_ball_x: float) -> float:
        base = (0.08, 0.16, 0.06, 0.05, 0.07, 0.1, 0.04, 0.26)[action]
        tactical = max(0.0, signed_ball_x) * 0.08
        fatigue = max(0.0, 0.52 - stamina) * 0.12
        return min(0.42, base + tactical + fatigue)

    @staticmethod
    def _rate(a: int | float, b: int | float) -> float:
        return round(float(a) / float(b), 3) if b else 0.0

    @staticmethod
    def _defensive_line_depth(action: int) -> float:
        return (0.38, 0.62, 0.24, 0.46, 0.42, 0.5, 0.18, 0.68)[action]

    @staticmethod
    def _pass_risk(action: int) -> float:
        return (0.35, 0.42, 0.18, 0.22, 0.72, 0.62, 0.12, 0.58)[action]

    @staticmethod
    def _pass_quality(action: int, stamina: float) -> float:
        base = (0.72, 0.64, 0.76, 0.84, 0.58, 0.62, 0.82, 0.55)[action]
        return base * (0.7 + stamina * 0.3)

    @staticmethod
    def _press_intensity(action: int, stamina: float) -> float:
        base = (0.45, 0.86, 0.28, 0.34, 0.38, 0.52, 0.2, 0.78)[action]
        return base * (0.65 + stamina * 0.35)

    @staticmethod
    def _foul_risk(action: int, stamina: float) -> float:
        base = (0.025, 0.055, 0.018, 0.02, 0.025, 0.038, 0.012, 0.09)[action]
        return min(0.22, base + max(0.0, 0.65 - stamina) * 0.055)

    def _progress(self, attack_action: int, defend_action: int) -> float:
        attack = (0.055, 0.045, 0.035, 0.06, 0.14, 0.12, 0.025, 0.105)[attack_action]
        defense = (0.0, 0.035, -0.025, -0.005, 0.0, 0.018, -0.035, 0.026)[defend_action]
        return max(-0.03, attack - defense + self.random.uniform(-0.018, 0.018))

    @staticmethod
    def _shot_chance(action: int, signed_ball_x: float, stamina: float) -> float:
        bias = (0.025, 0.018, 0.012, 0.02, 0.115, 0.085, 0.01, 0.075)[action]
        field = max(0.0, signed_ball_x) * 0.12
        return min(0.42, (bias + field) * (0.65 + stamina * 0.35))
