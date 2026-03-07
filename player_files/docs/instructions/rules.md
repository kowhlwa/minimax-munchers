# ByteFight 2026: Paint

In the distant digital future, a mysterious virus has wiped out the primordial cyber-serpents, leaving behind only the algorithmic energy upon which they feasted.

In the aftermath, desperate travelers from other universes began appearing, realizing that they could siphon computational power from this reality to stabilize their own decaying worlds. These "painters" now vie for control over conduits of power in the ashen ruins of this forgotten domain, tethered to their original reality only by the signal beacons they have brought with them. They know that success will ensure the survival of their existence, and failure will bring catastrophic consequences. This is **ByteFight 2026: Paint**.


## Objective

The goal of ByteFight 2026: Paint is to defeat your opponent’s agent by achieving strategic control of the battlefield.

A match can be won in any of the following ways:

1. **Stamina Collapse**: If your opponent’s stamina decreases below 0 at any point, their agent destabilizes and is immediately removed from the world. You win the match.

2. **Collision Resolution**: If both agents attempt to occupy the same cell and a collision occurs, the game resolves the interaction according to the collision rules described later. If you win the collision resolution, your opponent’s agent is destroyed and you win the match.

3. **Hill Dominance**: If you control more than 75% of all Hills on the map, your agent synchronizes the region with your home universe, forcing all opposing signals out of the domain. You win the match.

## Game Overview

ByteFight is a turn-based strategy game played on a grid map. Each player controls a single agent that moves around the map painting territory, capturing hills, and competing for resources.

Players spend a shared resource called stamina to perform actions such as:

- Moving
- Painting territory
- Erasing opponent paint
- Deploying beacons
- Taking multiple moves in a single turn

The player who manages stamina most effectively while controlling territory will gain strategic advantage.

## The Map

Each match takes place on a grid-based map with a size ranging from:

- 8×8
- up to 32×32

Maps are generated with one of the following symmetries:

- Vertical symmetry
- Horizontal symmetry
- Rotational symmetry

This ensures that both players start with fair map conditions.

## Cell Types

Each grid cell may contain one or more of the following elements.

### Walls

A Wall is an impassable tile.

- Agents cannot move into a wall.
- Nothing can spawn on a wall.

### Paint

A cell may contain paint from only one player at a time.
Paint represents territorial control and contributes to stamina regeneration.
Paint can stack up to 3 layers.

### Beacons

A cell may contain one beacon.
Beacons act as teleport anchors and count as controlled cells.
Only one player's beacon may occupy a cell.

### Hills

Some cells belong to a Hill region.
Controlling hills increases your maximum stamina.

### Powerups

Cells may contain powerups which restore stamina.
Powerups:

- Spawn symmetrically
- Appear at scheduled intervals
- Spawn locations are randomly chosen at the start of the match
  Powerups restore stamina but cannot exceed your maximum stamina.

## Controlling Cells

A cell is controlled by a player if:
The cell contains paint from that player
OR
The cell contains a beacon placed by that player
Control is used for:

- Hill capture
- Stamina regeneration

## Stamina System

Stamina is the primary resource in the game.

### What Stamina Is Used For

Stamina is consumed when performing actions such as:

- Painting
- Erasing paint
- Taking multiple moves in a turn
  If your stamina drops below 0, you immediately lose.

### Stamina Regeneration

Stamina regenerates each turn based on territory control.
Stamina gained each turn is calculated as:

- Base regeneration: 5 stamina
- Territory bonus: +1 stamina for every 8 cells painted
- Local control bonus: +2 stamina for each controlled cell within a 5×5 area centered on your agent

(The 5×5 area includes 25 cells around your agent.)

### Maximum Stamina

Maximum stamina limits the amount of stamina you may hold.
Max stamina increases by capturing Hills.

- Each hill controlled increases max stamina by 40

## Paint System

Painting territory is one of the core mechanics of the game.

### Painting Rules

When performing Action.Paint:

- The target cell must be Manhattan distance 1 from the player.

- You cannot paint your current cell.

- You cannot paint a cell that contains opponent paint.

- You cannot paint a cell containing a beacon.

### Paint Layers

Paint stacks up to 3 layers.

Layers provide durability against erasing.

### Paint Cost

Each paint action costs:

- 15 stamina

## Hills

Hills are special map regions that provide powerful strategic advantages.

### Capturing Hills

To control a hill, a player must:

- Control more hill cells than the opponent

- Control at least 33% of the hill’s cells

### Hill Benefits

Each controlled hill increases:

Maximum stamina by 40

Capturing enough hills can also trigger the Hill Dominance win condition.

## Beacons

Beacons act as control anchors and teleportation points.

### Deploying Beacons

A beacon can only be deployed if:

- 7 of the 9 cells in a 3×3 grid centered on the player are painted by that player

- Beacons do not count as painted cells

When deploying a beacon:

- The entire 3×3 paint region is consumed

- The beacon is placed at the player's position

### Beacon Teleportation

If a player is standing on one of their own beacons, they may:

Teleport to another beacon they own.

Effects:

- The destination beacon is consumed

- Teleportation replaces normal movement

## Actions

Each turn, your controller may return a list of actions.

You may perform multiple actions per turn, limited by stamina.

### Move Action

When playing Action.Move:

#### Regular Step

Move one tile in a cardinal direction:

- North

- South

- East

- West

If the target tile contains opponent paint:

- 1 layer is automatically erased

#### Erase Step

You may spend additional stamina to erase paint.

Cost:

50 stamina

Effect:

- All paint layers on the stepped tile are removed.

#### Beacon Step

If standing on your own beacon:

You may teleport to another beacon you own.

Effects:

- The destination beacon is destroyed.

#### Multiple Moves

You may perform additional moves in the same turn.

Each additional move costs increasing stamina:

- 2nd move: 10 stamina

- 3rd move: 20 stamina

- 4th move: 30 stamina

- etc.

## Paint Action

When playing `Action.Paint`:

- Paint a cell at Manhattan distance 1

- Max 3 layers

- Cost: 15 stamina

## Collisions

A collision occurs when both agents attempt to occupy the same cell.

Collision resolution follows these rules:

1. The player who controls the cell wins the collision.

2. If neither player controls the cell, the player who initiates the collision wins.

3. Erase steps cannot be used when initiating a collision.

The losing agent is immediately destroyed.

## Bid

Your agent must implement **two functions**: `bid` and `play`.

Before gameplay begins, you may **bid stamina** to get the **first move**.

## Turn System & Time Limits

Each match has strict compute limits.

Players receive:

- 10 seconds to initialize their PlayerController

- 20 seconds to bid stamina for initiative

- 180 seconds total compute time to play the game

Each match allows a maximum of:

- 1000 turns per player

- 2000 total turns

## Ranked Match Hardware

Ranked matches run on:

**ACEMAGIC M1 AMD Ryzen™ 7 6800H / 7735HS Mini PC**

Each agent receives:

- 3 threads

Hardware availability for scrimmages is **not guaranteed**.

## Sudden Death

After 1000 turns (500 rounds) the game enters Sudden Death.

During sudden death:

Every 100 moves, the base stamina regeneration decreases by 1.

This gradually forces the match toward a conclusion.

## Tiebreakers

If the game reaches 2000 turns (1000 rounds) without a winner, a tiebreak is used.

The winner is determined using the following metrics, in order:

1. Number of Hills Controlled

2. Number of Cells Controlled

3. Remaining Compute Time
