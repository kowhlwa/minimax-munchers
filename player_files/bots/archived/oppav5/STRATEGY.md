# oppav5 -> oppav6: Comprehensive Upgrade Strategy

## Executive Summary
Transform oppav5 from a strong hill-rusher into an adaptive, phase-aware competitive bot
that dominates skilled opponents across all map types and game stages.

## Architecture: 5-Phase Game Engine

Replace the monolithic play() with a phase-based system. Each phase has distinct priorities
but shares the same safety infrastructure.

### Phase 1: FARM (turns 0-N, dynamic exit)
**Trigger**: Game start on large maps, or when local regen < 6 cells in 5x5 radius
**Goal**: Build a territory base around spawn to sustain stamina regen
**Exit when**: local_friendly_cells >= 6 in 5x5 AND (stamina >= 60 OR nearest_hill_dist <= 3)
**Actions**:
- Paint aggressively in spiral/frontier pattern from spawn
- Take 2 moves per turn max (conserve stamina)
- Grab adjacent powerups
- Skip hill rushing until territory sustains regen
**Why**: bangv13 beats oppav5 on large maps because oppav5 rushes hills without
territory base, then starves. Regen-gating prevents stamina death.

### Phase 2: RUSH (primary phase)
**Trigger**: Exit from FARM, or game start on small/medium maps
**Goal**: Capture hills using MST-sequenced routing (existing oppav5 strength)
**Exit when**: All reachable hills captured OR opponent has clear advantage requiring defense
**Actions**:
- Existing MST hill sequence logic (improved with full BFS for all hops)
- Up to 4 moves on hills
- Paint-then-move for collision safety
- Kill detection (existing 1-3 step)
**Improvements over oppav5**:
- Full BFS for all MST hops instead of Manhattan for subsequent hops
- Better commitment logic: don't abandon contested hills too early

### Phase 3: DEFEND
**Trigger**: Opponent approaching our controlled hill (urgency >= 0.10, down from 0.15)
**Goal**: Protect threatened hills, reinforce paint layers
**Exit when**: Hill no longer threatened OR hill lost (switch to RUSH to recapture)
**Actions**:
- Rush to threatened hill
- Reinforce paint to 3-4 layers on critical cells
- Position on owned cells near hill edge to threaten collision
- Use chokepoint defense (actually implement what oppav5 detects but doesn't use)

### Phase 4: CONTEST
**Trigger**: Both players on same hill or within 3 cells of each other on a hill
**Goal**: Win the hill through superior positioning and painting
**Actions**:
- Edge-sweep strategy (attack from perimeter)
- Paint neutral cells aggressively
- Safe-step with collision awareness
- Pre-paint before stepping onto neutral cells in threat range
- Multi-layer reinforcement on won cells

### Phase 5: ENDGAME (turns 700+)
**Trigger**: Turn count > 700, or all hills controlled, or clear lead
**Goal**: Maximize territory for tiebreak, protect hill advantage
**Actions**:
- Shift to territory expansion (frontier targeting)
- Reinforce hill paint to max (4 layers)
- Maintain safe positioning
- Calculate: "do I win on tiebreak?" and play accordingly
  - If winning: play conservatively, don't risk collisions
  - If losing: aggressive territory grab, consider collision rush

---

## Improvement Modules (in priority order)

### 1. Regen-Gating (CRITICAL - prevents stamina death)
```
Before committing to any distant hill:
- Count local_friendly in 5x5 around current position
- If local_friendly < 6: enter FARM phase
- Only exit FARM when sustainable regen established
```
This single change fixes the biggest oppav5 weakness on large maps.

### 2. BFS Caching
```
Store last BFS result with a hash of (my_position, board_turn).
Reuse when position hasn't changed (e.g., during paint-only actions).
Cache opponent BFS separately (changes every turn).
```
Saves ~50% computation on turns where we paint from same position.

### 3. Smart Dynamic Bidding
```python
def bid(self, board, pp, time_left):
    me = board.get_player(pp)
    opp = board.get_opponent(pp)

    # Analyze closest hill and race advantage
    closest_hill_dist_me = ...  # BFS to nearest uncaptured hill
    closest_hill_dist_opp = ... # Manhattan estimate for opponent

    if closest_hill_dist_me <= 2 and closest_hill_dist_opp >= 4:
        return 15  # We're closer, bid to go first and capture

    if closest_hill_dist_me == closest_hill_dist_opp:
        return 12  # Contested, bid moderately

    if len(me.controlled_hills) > len(opp.controlled_hills):
        return 3   # We're ahead, save stamina

    return 6  # Default conservative
```

### 4. Opponent Pattern Tracking
```
Track per-turn:
- Opponent position history (last 16 turns)
- Direction trend (moving toward which hill?)
- Aggression level (approaching us vs. farming vs. rushing hills)
- Stamina trajectory (gaining or depleting?)

Use to:
- Predict which hill opponent is targeting
- Detect when opponent is farming (we should rush)
- Detect when opponent is rushing us (prepare collision defense)
- Time our engagements for when opponent is low stamina
```

### 5. Beacon Strategy (currently unused!)
oppav5 NEVER places or uses beacons. This is a massive untapped advantage.
```
Beacon placement:
- When 7/9 cells in 3x3 are painted and on our territory
- Place near hills we control for fast return
- Place near spawn for emergency retreat

Beacon travel:
- When defending a threatened hill, teleport from far beacon
- When opponent is between us and target, bypass via beacon
- Saves stamina compared to walking (consumes beacon but 0 stamina)
```

### 6. Multi-Move Lookahead (2-ply)
```
Before committing to move sequence:
- Simulate 2-3 possible first moves
- For each, evaluate resulting position:
  - Regen potential at new position
  - Collision safety
  - Progress toward target
  - Paint opportunities
- Pick the move sequence with best composite score
```
This replaces the current greedy single-step selection.

### 7. Improved Kill Detection
```
Current: Only checks 1-3 step kills
Enhanced:
- Check if opponent WILL be killable next turn (set up kills)
- Use forecast_turn() to simulate moves and check collision outcomes
- Multi-turn kill plans: paint a cell this turn, kill next turn
- Detect when opponent is walking into our territory (don't run away!)
```

### 8. Anti-Farming Counter
```
When opponent is farming (detected by pattern tracker):
- Rush uncontested hills immediately (they're not competing)
- Don't waste moves on territory near their farm zone
- Take 3-4 move turns to maximize hill acquisition speed
```

### 9. Territory Pressure Strategy
```
When ahead on hills, shift to territorial pressure:
- Paint toward opponent's base/spawn
- Deny opponent regen zones
- Force them into low-regen areas
- Paint thick (3-4 layers) to make erasure expensive for them
```

### 10. Late-Game Stamina Optimization
```
After turn 700:
- Calculate exact regen with decay
- If stamina will decay to 0 before turn 1000:
  - Rush for domination win while stamina exists
  - Or play ultra-conservative if ahead on territory
- Track exact tiebreak score and play accordingly
```

---

## Implementation Order

1. Phase system skeleton (wrapping existing code)
2. Regen-gating (FARM phase)
3. Smart bidding
4. Opponent pattern tracking
5. Improved MST (full BFS all hops)
6. Better defense triggers
7. Endgame strategy
8. Beacon strategy
9. Multi-move lookahead
10. Kill setup detection

Each improvement should be ADDITIVE - no removing existing working logic,
only wrapping it in smarter decision-making.
