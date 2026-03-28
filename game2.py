import pygame
import random
import heapq
import math
import time

W, H    = 900, 700
CELL    = 28
FPS     = 60

EMPTY   = 0
WALL    = 1
SPAWN   = 2
EXIT    = 3
OBJECT  = 4
ANOMALY = 5

C_BG       = (5,   5,   10)
C_WALL     = (14,  12,  22)
C_WALL_LIT = (28,  24,  44)
C_FLOOR    = (16,  14,  28)
C_PATH     = (20,  50,  36)
C_SPAWN    = (0,   220, 180)
C_EXIT     = (220, 30,  55)
C_OBJECT   = (120, 60,  20)
C_ANOMALY  = (220, 100, 0)
C_PLAYER   = (255, 250, 220)
C_FOG      = (5,   5,   10)
C_TEXT     = (200, 195, 215)
C_ACCENT   = (220, 30,  55)
C_GREEN    = (0,   220, 180)
C_GOLD     = (200, 170, 60)
C_DIM      = (80,  75,  100)
C_PANEL    = (10,  9,   18)
C_PANELBDR = (40,  30,  60)

# rows, cols, extra_wall_open%, objects, anomalies, fog_radius, time_limit, name
LEVEL_CONFIGS = [
    dict(rows=13, cols=17, extra=0.05, objects=3,  anomalies=3,  fog=7, time=0,   name="The Hallway"),
    dict(rows=15, cols=20, extra=0.10, objects=5,  anomalies=5,  fog=6, time=75,  name="The Basement"),
    dict(rows=17, cols=22, extra=0.15, objects=7,  anomalies=7,  fog=5, time=90,  name="The Attic"),
    dict(rows=19, cols=24, extra=0.20, objects=9,  anomalies=10, fog=4, time=110, name="The Catacombs"),
    dict(rows=21, cols=27, extra=0.25, objects=12, anomalies=14, fog=3, time=130, name="The Void"),
]

ANOMALY_MSGS = [
    "mirror shattered", "candle snuffed", "figure vanished",
    "door sealed", "painting moved", "clock reversed",
    "doll displaced", "shadow lingers", "whisper heard",
    "furniture shifted", "window fogged", "book opened",
    "footsteps above", "light flickered", "cold breath felt",
]

# UTILITIES
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def cell_key(r, c):
    return r * 1000 + c

# A* PATHFINDING
def astar(grid, start, goal):
    R, C = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    def h(r, c):
        return abs(r - gr) + abs(c - gc)

    open_heap = [(h(sr, sc), 0, sr, sc)]
    g_cost = {cell_key(sr, sc): 0}
    parent = {}

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        if r == gr and c == gc:
            path, k = [], cell_key(r, c)
            while k in parent:
                path.append(k)
                k = parent[k]
            path.append(cell_key(sr, sc))
            path.reverse()
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr][nc] == WALL:
                continue
            ng = g + 1
            nk = cell_key(nr, nc)
            if nk not in g_cost or ng < g_cost[nk]:
                g_cost[nk] = ng
                parent[nk] = cell_key(r, c)
                heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))
    return []


# LEVEL GENERATION  — imperfect maze = multiple paths
def generate_grid(rows, cols, extra_open_pct):
    g = [[WALL] * cols for _ in range(rows)]

    # DFS maze carver
    visited = [[False] * cols for _ in range(rows)]
    stack = [(1, 1)]
    g[1][1] = EMPTY
    visited[1][1] = True
    while stack:
        r, c = stack[-1]
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        random.shuffle(dirs)
        moved = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows-1 and 0 < nc < cols-1 and not visited[nr][nc]:
                g[r + dr//2][c + dc//2] = EMPTY
                g[nr][nc] = EMPTY
                visited[nr][nc] = True
                stack.append((nr, nc))
                moved = True
                break
        if not moved:
            stack.pop()

    # Open extra walls to create loops / multiple paths
    extra_target = int(rows * cols * extra_open_pct * 0.4)
    wall_cells = [
        (r, c) for r in range(1, rows-1) for c in range(1, cols-1)
        if g[r][c] == WALL
    ]
    random.shuffle(wall_cells)
    opened = 0
    for r, c in wall_cells:
        if opened >= extra_target:
            break
        horiz = (0 < c < cols-1) and g[r][c-1] == EMPTY and g[r][c+1] == EMPTY
        vert  = (0 < r < rows-1) and g[r-1][c] == EMPTY and g[r+1][c] == EMPTY
        if horiz or vert:
            g[r][c] = EMPTY
            opened += 1

    # Mid-map horizontal blockage — forces the player to find another route
    mid_r  = rows // 2
    s_col  = random.randint(cols // 4, cols // 2)
    seg    = cols // 5
    for c in range(s_col, min(s_col + seg, cols - 1)):
        if g[mid_r][c] == EMPTY:
            g[mid_r][c] = WALL

    # Vertical blockage near 3/4 point
    mid_c  = (cols * 3) // 4
    s_row  = random.randint(rows // 4, rows // 2)
    vseg   = rows // 5
    for r in range(s_row, min(s_row + vseg, rows - 1)):
        if g[r][mid_c] == EMPTY:
            g[r][mid_c] = WALL

    g[1][1]            = SPAWN
    g[rows-2][cols-2]  = EXIT
    return g


def place_objects(g, rows, cols, n):
    placed, tries = 0, 0
    while placed < n and tries < 600:
        tries += 1
        r, c = random.randint(1, rows-2), random.randint(1, cols-2)
        if g[r][c] == EMPTY:
            g[r][c] = OBJECT
            placed += 1


def inject_anomalies(g, rows, cols, n, path):
    path_set = set(path)
    log, injected, tries = [], 0, 0
    while injected < n and tries < 800:
        tries += 1
        r, c = random.randint(1, rows-2), random.randint(1, cols-2)
        k = cell_key(r, c)
        if g[r][c] in (EMPTY, OBJECT) and k not in path_set:
            old = g[r][c]
            g[r][c] = ANOMALY
            if astar(g, (1, 1), (rows-2, cols-2)):
                injected += 1
                log.append("  " + random.choice(ANOMALY_MSGS))
            else:
                g[r][c] = old
    return log

# FOG OF WAR — Bresenham LOS per cell
def compute_visible(grid, pr, pc, radius):
    rows, cols = len(grid), len(grid[0])
    vis = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr*dr + dc*dc > radius*radius:
                continue
            tr, tc = pr + dr, pc + dc
            if not (0 <= tr < rows and 0 <= tc < cols):
                continue
            # Bresenham trace from player to target
            x0, y0 = pc, pr
            x1, y1 = tc, tr
            dx = abs(x1-x0); sx = 1 if x0 < x1 else -1
            dy = -abs(y1-y0); sy = 1 if y0 < y1 else -1
            err = dx + dy
            cx, cy = x0, y0
            blocked = False
            while True:
                if blocked:
                    break
                vis.add((cy, cx))
                if cx == x1 and cy == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy; cx += sx
                if e2 <= dx:
                    err += dx; cy += sy
                if grid[cy][cx] == WALL:
                    vis.add((cy, cx))
                    blocked = True
    return vis
# DRAW HELPERS
def draw_text(surf, text, font, color, x, y, anchor="topleft"):
    s = font.render(text, True, color)
    r = s.get_rect(**{anchor: (x, y)})
    surf.blit(s, r)
    return r


def draw_cell(surf, grid, r, c, ox, oy, path_set, visible, explored, anim_t):
    x, y = ox + c * CELL, oy + r * CELL
    cell  = grid[r][c]
    k     = cell_key(r, c)
    is_vis = (r, c) in visible
    is_exp = (r, c) in explored

    if not is_vis and not is_exp:
        pygame.draw.rect(surf, C_FOG, (x, y, CELL, CELL))
        return

    dim = 1.0 if is_vis else 0.22

    if cell == WALL:
        col = lerp_color(C_WALL, C_WALL_LIT, 0.5 * dim)
        pygame.draw.rect(surf, col, (x, y, CELL, CELL))
        edge = lerp_color(col, (0, 0, 0), 0.5)
        pygame.draw.rect(surf, edge, (x+1, y+1, CELL-2, CELL-2), 1)
    else:
        on_path = k in path_set and is_vis
        base = lerp_color(C_FLOOR, C_PATH, 0.55) if on_path else C_FLOOR
        base = lerp_color(C_FOG, base, dim)
        pygame.draw.rect(surf, base, (x, y, CELL, CELL))
        pygame.draw.rect(surf, lerp_color(base, (255, 255, 255), 0.04),
                         (x, y, CELL, CELL), 1)

        if not is_vis:
            return

        cx, cy = x + CELL // 2, y + CELL // 2

        if cell == SPAWN:
            pulse = 0.5 + 0.5 * math.sin(anim_t * 3)
            pygame.draw.rect(surf, lerp_color(C_BG, C_SPAWN, 0.3 * pulse),
                             (x+2, y+2, CELL-4, CELL-4), border_radius=4)
            pygame.draw.rect(surf, C_SPAWN, (x+2, y+2, CELL-4, CELL-4), 2, border_radius=4)

        elif cell == EXIT:
            pulse = 0.5 + 0.5 * math.sin(anim_t * 4.5)
            pygame.draw.rect(surf, lerp_color(C_BG, C_EXIT, 0.4 * pulse),
                             (x+2, y+2, CELL-4, CELL-4), border_radius=4)
            pygame.draw.rect(surf, C_EXIT, (x+2, y+2, CELL-4, CELL-4), 2, border_radius=4)

        elif cell == OBJECT:
            pygame.draw.rect(surf, (45, 22, 5), (x+4, y+4, CELL-8, CELL-8), border_radius=2)
            pygame.draw.rect(surf, C_OBJECT, (x+4, y+4, CELL-8, CELL-8), 1, border_radius=2)

        elif cell == ANOMALY:
            pulse = 0.5 + 0.5 * math.sin(anim_t * 7)
            pygame.draw.rect(surf, lerp_color(C_BG, C_ANOMALY, 0.25 * pulse),
                             (x+2, y+2, CELL-4, CELL-4), border_radius=3)
            pygame.draw.rect(surf, C_ANOMALY, (x+2, y+2, CELL-4, CELL-4), 2, border_radius=3)


def draw_player(surf, pr, pc, ox, oy, anim_t):
    cx = ox + pc * CELL + CELL // 2
    cy = oy + pr * CELL + CELL // 2
    rad = CELL // 2 - 4
    # Outer glow rings
    for i in range(3, 0, -1):
        a = 40 - i * 10
        gsurf = pygame.Surface((rad*2 + i*8, rad*2 + i*8), pygame.SRCALPHA)
        pygame.draw.circle(gsurf, (*C_PLAYER, a),
                           (rad + i*4, rad + i*4), rad + i*4)
        surf.blit(gsurf, (cx - rad - i*4, cy - rad - i*4))
    pygame.draw.circle(surf, C_PLAYER, (cx, cy), rad)
    bob = int(math.sin(anim_t * 5) * 2)
    pygame.draw.circle(surf, C_BG, (cx, cy + bob), rad // 3)


def draw_hud(surf, fonts, player_name, level_num, level_name,
             elapsed, time_limit, anomaly_cnt, log_lines):
    PANEL_W = 188
    pygame.draw.rect(surf, C_PANEL, (W - PANEL_W, 0, PANEL_W, H))
    pygame.draw.line(surf, C_PANELBDR, (W-PANEL_W, 0), (W-PANEL_W, H), 1)

    px = W - PANEL_W + 10
    py = 14

    def label(txt):
        nonlocal py
        draw_text(surf, txt, fonts['tiny'], C_DIM, px, py)
        py += 15

    def value(txt, col=C_TEXT):
        nonlocal py
        draw_text(surf, txt, fonts['body'], col, px, py)
        py += 28

    def sep():
        nonlocal py
        pygame.draw.line(surf, C_PANELBDR, (px, py), (W-10, py), 1)
        py += 8

    label("SURVIVOR")
    value(player_name[:14], C_GREEN)
    sep()
    label("LEVEL")
    draw_text(surf, str(level_num), fonts['body'], C_ACCENT, px, py)
    draw_text(surf, level_name, fonts['tiny'], C_DIM, px+26, py+6)
    py += 28
    sep()
    label("TIME")
    if time_limit > 0:
        left = max(0, time_limit - elapsed)
        col  = C_ACCENT if left < 20 else (C_GOLD if left < 40 else C_TEXT)
        value(f"{left}s remaining", col)
        # Timer bar
        bw = PANEL_W - 20
        pygame.draw.rect(surf, C_PANELBDR, (px, py, bw, 5), border_radius=2)
        fill = int(bw * left / time_limit)
        if fill > 0:
            pygame.draw.rect(surf, col, (px, py, fill, 5), border_radius=2)
        py += 14
    else:
        value(f"{elapsed}s elapsed")
    sep()
    label("ANOMALIES NEARBY")
    value(str(anomaly_cnt), C_ANOMALY)
    sep()

    # Legend
    label("LEGEND")
    items = [
        (C_SPAWN,   "S  Spawn"),
        (C_EXIT,    "E  Exit"),
        (C_PLAYER,  "P  You"),
        (C_ANOMALY, "!  Anomaly"),
        (C_OBJECT,  "O  Object"),
        (C_PATH,    "   A* Path"),
    ]
    for col, txt in items:
        pygame.draw.rect(surf, col, (px, py+2, 10, 10), border_radius=2)
        draw_text(surf, txt, fonts['tiny'], C_DIM, px+14, py+1)
        py += 15
    sep()

    label("CONTROLS")
    for line in ["Arrows / WASD  Move", "R              Restart", "ESC            Quit"]:
        draw_text(surf, line, fonts['tiny'], C_DIM, px, py)
        py += 13
    sep()

    label("ANOMALY LOG")
    for line in log_lines[-10:]:
        if py + 13 > H - 6:
            break
        draw_text(surf, line, fonts['tiny'],
                  lerp_color(C_ANOMALY, C_DIM, 0.5), px, py)
        py += 13


def draw_intro(surf, fonts, player_name, cursor_on, error_msg, anim_t):
    surf.fill(C_BG)

    # Blood drips across top
    rng = random.Random(42)
    for i in range(W // 18):
        dh = rng.randint(18, 65)
        pygame.draw.rect(surf, (75, 0, 0), (i*18 + rng.randint(-2,2), 0, 5, dh))

    # Title glow
    for i in range(4, 0, -1):
        gs = fonts['title'].render("PhantomPath", True,
                                   lerp_color(C_ACCENT, C_BG, 1 - i / 5))
        r = gs.get_rect(center=(W//2, 140))
        surf.blit(gs, r.move(random.randint(-1, 1), random.randint(-1, 1)))
    t = fonts['title'].render("PhantomPath", True, C_ACCENT)
    surf.blit(t, t.get_rect(center=(W//2, 140)))

    sub = fonts['small'].render(
        "PROCEDURAL HORROR  ·  A* VALIDATION  ·  ANOMALY INJECTION",
        True, C_DIM)
    surf.blit(sub, sub.get_rect(center=(W//2, 188)))

    sk_alpha = 0.3 + 0.4 * math.sin(anim_t)
    sk = fonts['body'].render("  ".join(["☠"] * 5), True,
                               lerp_color(C_BG, C_DIM, sk_alpha))
    surf.blit(sk, sk.get_rect(center=(W//2, 228)))

    # Input
    bw, bh = 360, 52
    bx, by = W//2 - bw//2, 285
    lbl = fonts['small'].render("WHAT IS YOUR NAME, WANDERER?", True, C_DIM)
    surf.blit(lbl, lbl.get_rect(center=(W//2, by - 22)))
    pygame.draw.rect(surf, C_PANEL, (bx, by, bw, bh), border_radius=4)
    bdr = C_GREEN if cursor_on else C_ACCENT
    pygame.draw.rect(surf, bdr, (bx, by, bw, bh), 2, border_radius=4)
    disp = player_name + ("|" if cursor_on else "")
    ns = fonts['hud'].render(disp, True, C_TEXT)
    surf.blit(ns, ns.get_rect(center=(W//2, by + bh//2)))

    if error_msg:
        es = fonts['small'].render(error_msg, True, C_ACCENT)
        surf.blit(es, es.get_rect(center=(W//2, by + bh + 18)))

    # Enter button
    bw2, bh2 = 260, 50
    bx2 = W//2 - bw2//2
    by2 = by + bh + (48 if not error_msg else 60)
    pulse = 0.5 + 0.5 * math.sin(anim_t * 2)
    bc = lerp_color(C_ACCENT, (255, 70, 90), pulse)
    pygame.draw.rect(surf, bc, (bx2, by2, bw2, bh2), border_radius=6)
    bt = fonts['body'].render("ENTER THE DARK", True, C_BG)
    surf.blit(bt, bt.get_rect(center=(W//2, by2 + bh2//2)))

    warn_alpha = 0.25 + 0.25 * math.sin(anim_t * 0.7)
    warn = fonts['tiny'].render("those who enter rarely return",
                                True, lerp_color(C_BG, C_DIM, warn_alpha))
    surf.blit(warn, warn.get_rect(center=(W//2, H - 28)))

    return pygame.Rect(bx2, by2, bw2, bh2)


def draw_level_banner(surf, fonts, level_num, level_name, alpha):
    ov = pygame.Surface((W, H), pygame.SRCALPHA)
    ov.fill((5, 5, 10, int(alpha * 215)))
    surf.blit(ov, (0, 0))
    t = fonts['title'].render(f"LEVEL  {level_num}", True, C_ACCENT)
    surf.blit(t, t.get_rect(center=(W//2, H//2 - 55)))
    n = fonts['body'].render(level_name, True, C_DIM)
    surf.blit(n, n.get_rect(center=(W//2, H//2 + 8)))


def draw_overlay(surf, fonts, win, player_name, elapsed, final=False):
    ov = pygame.Surface((W, H), pygame.SRCALPHA)
    ov.fill((5, 5, 10, 210))
    surf.blit(ov, (0, 0))

    if final:
        title, sub_txt, col, btxt = (
            "YOU SURVIVED",
            f"{player_name} conquered all 5 levels. The darkness retreats.",
            C_GREEN, "PLAY AGAIN"
        )
    elif win:
        title, sub_txt, col, btxt = (
            "YOU ESCAPED",
            f"Cleared in {elapsed}s.  The darkness grows deeper…",
            C_GREEN, "DESCEND DEEPER  →"
        )
    else:
        title, sub_txt, col, btxt = (
            "CONSUMED",
            f"{player_name} was swallowed by the dark.",
            C_ACCENT, "TRY AGAIN"
        )

    t = fonts['title'].render(title, True, col)
    surf.blit(t, t.get_rect(center=(W//2, H//2 - 85)))
    s = fonts['body'].render(sub_txt, True, C_DIM)
    surf.blit(s, s.get_rect(center=(W//2, H//2)))

    bw, bh = 310, 50
    bx, by = W//2 - bw//2, H//2 + 55
    pygame.draw.rect(surf, col, (bx, by, bw, bh), border_radius=6)
    bt = fonts['body'].render(btxt, True, C_BG)
    surf.blit(bt, bt.get_rect(center=(W//2, by + bh//2)))
    return pygame.Rect(bx, by, bw, bh)

# MAIN GAME CLASS
class PhantomPath:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("PhantomPath AI")
        self.clock  = pygame.time.Clock()

        self.fonts = {
            'title': pygame.font.SysFont("impact",     60),
            'body':  pygame.font.SysFont("impact",     26),
            'hud':   pygame.font.SysFont("couriernew", 21, bold=True),
            'small': pygame.font.SysFont("couriernew", 12, bold=True),
            'tiny':  pygame.font.SysFont("couriernew", 11),
        }

        self.state        = "intro"
        self.player_name  = ""
        self.error_msg    = ""
        self.current_lvl  = 0
        self.grid         = []
        self.rows = self.cols = 0
        self.ox = self.oy = 0
        self.player       = (1, 1)
        self.path         = []
        self.path_set     = set()
        self.visible      = set()
        self.explored     = set()
        self.anomaly_log  = []
        self.anomaly_cnt  = 0
        self.start_time   = 0.0
        self.elapsed      = 0
        self.game_active  = False
        self.overlay_win  = False
        self.final_win    = False
        self.overlay_btn  = None
        self.intro_btn    = None
        self.banner_timer = 0.0
        self.cursor_on    = True
        self.cursor_timer = 0.0
        self.anim_t       = 0.0

    def load_level(self):
        self.game_active = False
        self.explored    = set()
        self.anomaly_log = []
        self.anomaly_cnt = 0
        self.elapsed     = 0
        self.player      = (1, 1)

        cfg       = LEVEL_CONFIGS[min(self.current_lvl, len(LEVEL_CONFIGS)-1)]
        self.rows = cfg['rows']
        self.cols = cfg['cols']

        play_w  = W - 190
        self.ox = max(0, (play_w - self.cols * CELL) // 2)
        self.oy = max(0, (H      - self.rows * CELL) // 2)

        # Generate until A* finds a valid path (should almost always be first try)
        for _ in range(300):
            self.grid = generate_grid(self.rows, self.cols, cfg['extra'])
            self.path = astar(self.grid, (1, 1), (self.rows-2, self.cols-2))
            if self.path:
                break

        self.path_set = set(self.path)
        place_objects(self.grid, self.rows, self.cols, cfg['objects'])
        log = inject_anomalies(self.grid, self.rows, self.cols,
                               cfg['anomalies'], self.path)
        self.anomaly_log.extend(log)
        self.anomaly_cnt = len(log)

        self._update_vis()
        self.start_time  = time.time()
        self.game_active = True
        self.state       = "banner"
        self.banner_timer = 2.2

    def _update_vis(self):
        cfg = LEVEL_CONFIGS[min(self.current_lvl, len(LEVEL_CONFIGS)-1)]
        pr, pc = self.player
        self.visible  = compute_visible(self.grid, pr, pc, cfg['fog'])
        self.explored |= self.visible


    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.anim_t       += dt
            self.cursor_timer += dt
            if self.cursor_timer > 0.5:
                self.cursor_on    = not self.cursor_on
                self.cursor_timer = 0.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                running = self._handle(event, running)

            self._update(dt)
            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _handle(self, event, running):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False

            if self.state == "intro":
                if event.key == pygame.K_RETURN:
                    self._try_start()
                elif event.key == pygame.K_BACKSPACE:
                    self.player_name = self.player_name[:-1]
                    self.error_msg   = ""
                elif event.unicode.isprintable() and len(self.player_name) < 18:
                    self.player_name += event.unicode
                    self.error_msg   = ""

            elif self.state == "game" and self.game_active:
                moves = {
                    pygame.K_UP: (-1,0),  pygame.K_w: (-1,0),
                    pygame.K_DOWN:(1,0),  pygame.K_s:  (1,0),
                    pygame.K_LEFT:(0,-1), pygame.K_a:  (0,-1),
                    pygame.K_RIGHT:(0,1), pygame.K_d:  (0,1),
                }
                if event.key in moves:
                    self._move(*moves[event.key])
                if event.key == pygame.K_r:
                    self.load_level()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.state == "intro" and self.intro_btn:
                if self.intro_btn.collidepoint(event.pos):
                    self._try_start()
            elif self.state == "overlay" and self.overlay_btn:
                if self.overlay_btn.collidepoint(event.pos):
                    self._overlay_action()

        return running

    def _try_start(self):
        name = self.player_name.strip()
        if not name:
            self.error_msg = "enter a name first!"
            return
        self.player_name = name
        self.current_lvl = 0
        self.load_level()

    def _move(self, dr, dc):
        pr, pc = self.player
        nr, nc = pr + dr, pc + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return
        if self.grid[nr][nc] == WALL:
            return
        if self.grid[nr][nc] == ANOMALY:
            self.anomaly_log.append("  you stepped on something…")
        self.player = (nr, nc)
        self._update_vis()
        if self.grid[nr][nc] == EXIT:
            self.elapsed     = int(time.time() - self.start_time)
            self.game_active = False
            self.overlay_win = True
            self.final_win   = (self.current_lvl >= len(LEVEL_CONFIGS) - 1)
            self.state       = "overlay"

    def _overlay_action(self):
        if self.final_win:
            self.state       = "intro"
            self.player_name = ""
            self.current_lvl = 0
        elif self.overlay_win:
            self.current_lvl += 1
            self.load_level()
        else:
            self.load_level()

    def _update(self, dt):
        if self.state == "banner":
            self.banner_timer -= dt
            if self.banner_timer <= 0:
                self.state = "game"

        if self.state == "game" and self.game_active:
            self.elapsed = int(time.time() - self.start_time)
            cfg = LEVEL_CONFIGS[min(self.current_lvl, len(LEVEL_CONFIGS)-1)]
            if cfg['time'] > 0 and self.elapsed >= cfg['time']:
                self.game_active = False
                self.overlay_win = False
                self.final_win   = False
                self.state       = "overlay"

    def _draw(self):
        self.screen.fill(C_BG)

        if self.state == "intro":
            self.intro_btn = draw_intro(
                self.screen, self.fonts,
                self.player_name, self.cursor_on,
                self.error_msg, self.anim_t
            )

        elif self.state in ("game", "banner"):
            self._draw_game()
            if self.state == "banner":
                cfg   = LEVEL_CONFIGS[min(self.current_lvl, len(LEVEL_CONFIGS)-1)]
                alpha = clamp(self.banner_timer / 2.0, 0, 1)
                draw_level_banner(self.screen, self.fonts,
                                  self.current_lvl + 1, cfg['name'], alpha)

        elif self.state == "overlay":
            self._draw_game()
            self.overlay_btn = draw_overlay(
                self.screen, self.fonts,
                self.overlay_win, self.player_name,
                self.elapsed, self.final_win
            )

    def _draw_game(self):
        cfg = LEVEL_CONFIGS[min(self.current_lvl, len(LEVEL_CONFIGS)-1)]

        for r in range(self.rows):
            for c in range(self.cols):
                draw_cell(
                    self.screen, self.grid, r, c,
                    self.ox, self.oy,
                    self.path_set, self.visible, self.explored,
                    self.anim_t
                )

        draw_player(self.screen, *self.player, self.ox, self.oy, self.anim_t)

        draw_hud(
            self.screen, self.fonts,
            self.player_name, self.current_lvl + 1, cfg['name'],
            self.elapsed, cfg['time'], self.anomaly_cnt,
            self.anomaly_log
        )

        # Red pulse bar when adjacent to anomaly
        pr, pc = self.player
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = pr+dr, pc+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] == ANOMALY:
                    p = 0.5 + 0.5 * math.sin(self.anim_t * 9)
                    bar = pygame.Surface((W, 4), pygame.SRCALPHA)
                    bar.fill((*C_ACCENT, int(200 * p)))
                    self.screen.blit(bar, (0, H - 4))
                    break

if __name__ == "__main__":
    PhantomPath().run()