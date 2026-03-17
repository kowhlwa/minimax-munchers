# ByteFight 2026: Paint

In the distant digital future, a mysterious virus has wiped out the primordial cyber-serpents, leaving behind only the algorithmic energy upon which they feasted.

In the aftermath, desperate travelers from other universes began appearing, realizing that they could siphon computational power from this reality to stabilize their own decaying worlds. These "painters" now vie for control over conduits of power in the ashen ruins of this forgotten domain, tethered to their original reality only by the signal beacons they have brought with them. They know that success will ensure the survival of their existence, and failure will bring catastrophic consequences. This is **ByteFight 2026: Paint**.

## Objective

The goal of ByteFight 2026: Paint is to defeat your opponent’s agent and achieve strategic control of the battlefield.

A match can be won in any of the following ways:

1. **Stamina Collapse**: If your opponent’s stamina decreases below 0 at any point, their agent destabilizes and is immediately removed from the world. You win the match.

2. **Collision Resolution**: If both agents attempt to occupy the same cell and a collision occurs, the game resolves the interaction according to the collision rules described later. If you win the collision resolution, your opponent’s agent is destroyed and you win the match.

3. **Hill Dominance**: If you control more than 75% of all Hills on the map, your agent synchronizes the region with your home universe, forcing all opposing signals out of the domain. You win the match.

## Game Overview

ByteFight is a turn-based strategy game played on a grid map. Each player controls a single agent that moves around the map painting territory, capturing hills, and competing for resources.

![Sample match board](/compete-images/sample_match.png)

Player agent sprites:

![Player 1 agent](/compete-images/player1.png)
![Player 2 agent](/compete-images/player2.png)

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
Paint can stack up to 4 layers.

### Beacons

A cell may contain one beacon.
Beacons act as teleport anchors and count as controlled cells.
Only one player's beacon may occupy a cell.

![Blue beacon example](/compete-images/blue_beacon.png)
![Green beacon example](/compete-images/green_beacon.png)

### Hills

Some cells belong to a Hill region.
Controlling hills increases your maximum stamina.

![Hill region example](/compete-images/hill.png)

### Powerups

Cells may contain powerups which restore stamina.
Powerups:

- Spawn symmetrically
- Appear at scheduled intervals
- Spawn locations are randomly chosen at the start of the match
  Powerups restore stamina but cannot exceed your maximum stamina.

![Power cell example](/compete-images/power_cell.png)

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
- If your stamina drops below 0, you immediately lose.

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

- The target cell must be manhattan distance 1 away from you.

- You cannot paint your current cell.

- You cannot paint a cell that contains opponent paint.

- You cannot paint a cell containing a beacon.

### Paint Layers

Paint stacks up to 4 layers.

Layers provide durability against erasing.

### Paint Cost

Each paint action costs:

- 15 stamina

## Hills

Hills are special map regions that provide powerful strategic advantages. Any cells denoted with hill ID 0 do not have hills in them.

### Capturing Hills

To control a hill, a player must:

- Control more hill cells than the opponent

- Control at least 50% of the hill’s cells

### Hill Control

Players will only lose control of a hill if their opponent manages to paint more of a hill than they do. See below for some scenarios:

- If an opponent erases one square of a 50% controlled 10-square hill and paints once (raising their control percent to 10% compared to the player's 40%), the player will not lose control of a hill

- If an opponent erases one square of a 50% controlled 10-square hill and paints 4 squares (raising their control percent to 40% compared to the player's 40%), the player will lose control of a hill but the opponent will not gain control.

- If an opponent erases one square of a 50% controlled 10-square hill and paints 5 squares (raising their control percent to 50% compared to the player's 40%), the player will lose control of a hill and the opponent will gain control of the hill.

- If a player places a beacon and a 10-square hill they previously controlled as a result is only painted with 2 tiles, the opponent need only paint 2 squares to cause them to lose control, but will need to paint 5 squares to gain control themselves.

### Hill Benefits

Each controlled hill increases:

- Maximum stamina by 40

- Capturing enough hills can also trigger the Hill Dominance win condition.

## Beacons

Beacons act as control anchors and teleportation points.

### Deploying Beacons

A beacon can only be deployed if:

- 7 of the 9 cells in a 3×3 grid centered on the player are painted by that player

- Beacons do not count as painted cells

Beacons can not be destroyed in any other way then teleporting from another beacon to that beacon.

When deploying a beacon:

- The 1 layer of the player's paint in the entire 3×3 paint region is consumed

- The beacon is placed at the player's position

- Any opponent paint in the 3x3 region is increased by 1

### Beacon Teleportation

If a player is standing on one of their own beacons, they may teleport to another beacon they own.

Effects:

- The destination beacon is consumed

- One layer paint is left behind after the beacon is consumed.

- Teleportation replaces normal movement

## Actions

Each turn, your controller may return a list of actions. Any of these actions can be made multiple times in the same turn. An Action.Move Step must be made in a turn. This includes Regular Step, Erase Step, or Beacon Step. Apart from this, you can also make as many Action.Paint steps as you want in a turn.

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

- The destination beacon is destroyed as mentioned before.

#### Multiple Moves

You may perform additional moves in the same turn. Moves refer to any Action.Move step (Regular, Erase, or Beacon)

Each additional move costs increasing stamina:

- 2nd move: 10 stamina

- 3rd move: 20 stamina

- 4th move: 30 stamina

- etc.

Here are a few examples. 
- If you use Beacon Step in your first step and teleport and then use Erase, Erase would cost 60 stamina.
- If you use an Erase Step (using 50 stamina) and then use another Erase Step, the second Erase Step would cost 60 stamina, in total costing 110 stamina.
- If you use a Regular Step and then another Regular step and then a Beacon Step, the first Regular Step would cost 0 stamina, the second would cost 10 stamina, and the Beacons step would cost 20 stamina for a total of 30 stamina used.

### Paint Action

When playing `Action.Paint`:

- Paint a cell at manhattan distance 1 (you cannot paint on top of yourself!)

- Max 4 layers

- Cost: 15 stamina

- You may paint multiple times in the same turn, either at the same cell or different cells. All of them will still cost 15 stamina.

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

Every 100 moves, the base stamina regeneration decreases by 1. This can cause base stamina regeneration to become negative, but cannot cause overall stamina regeneration to become negative (overall stamina regeneration is constrained to be at least 0.)

## Tiebreakers

If the game reaches 2000 turns (1000 rounds) without a winner, a tiebreak is used.

The winner is determined using the following metrics, in order:

1. Number of Hills Controlled

2. Number of Cells Controlled

3. Remaining Compute Time
