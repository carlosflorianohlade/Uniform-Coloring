import cv2
import numpy as np
import tensorflow as tf
from aima import Problem, Node, memoize, PriorityQueue
from dataclasses import dataclass
from collections.abc import Callable
import os
import matplotlib.pyplot as plt
import time
import imageio.v2 as imageio

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

MODEL_PATH = 'cnn_coloring_project.keras'
IMAGE_PATH = 'images/immagine2.jpeg'

# Mappa classi: indice output CNN -> lettera colore
CLASS_MAP = {0: 'B', 1: 'G', 2: 'T', 3: 'Y'}

# =============================================================================
# LOGICA DI VISIONE
# =============================================================================

def cluster_positions(positions, threshold=10):
    """Raggruppa coordinate vicine per evitare doppie righe o colonne."""
    clustered = []
    for p in sorted(positions):
        if not clustered or abs(p - clustered[-1]) > threshold:
            clustered.append(p)
    return clustered


def auto_crop_cell(img_array):
    """
    Trova il contorno della lettera, lo centra su canvas quadrato
    con padding e ridimensiona a 28x28 (formato EMNIST).
    """
    # Converti in scala di grigi se necessario
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array.copy()

    # Binarizzazione con Otsu invertita (lettera nera su bianco -> bianca su nero)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # se non ci sono contorni, restituiamo l'immagine binarizzata ridimensionata
    if not contours:
        return cv2.resize(binary, (28, 28))

    # Prendi il bounding box del contorno più grande
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crea canvas quadrato con padding del 20%
    square_size = max(w, h)
    padding     = int(square_size * 0.2)
    square_size += 2 * padding

    center_x = x + w // 2
    center_y = y + h // 2
    half     = square_size // 2

    canvas = np.zeros((square_size, square_size), dtype=np.uint8)

    # Coordinate sorgente (clampate ai bordi dell'immagine)
    sx0 = max(center_x - half, 0)
    sy0 = max(center_y - half, 0)
    sx1 = min(center_x + half, binary.shape[1])
    sy1 = min(center_y + half, binary.shape[0])

    # Coordinate destinazione sul canvas
    dx0 = max(0, half - (center_x - sx0))
    dy0 = max(0, half - (center_y - sy0))
    crop = binary[sy0:sy1, sx0:sx1]

    # controllo esplicito delle dimensioni prima di scrivere sul canvas
    dh, dw = crop.shape
    if dy0 + dh <= canvas.shape[0] and dx0 + dw <= canvas.shape[1]:
        canvas[dy0:dy0+dh, dx0:dx0+dw] = crop

    return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)


def extract_grid_to_matrix(image_path, model, verbose=False):
    """
    Legge l'immagine, rileva la griglia, classifica ogni cella con la CNN
    e restituisce (tupla_1D, righe, colonne).
    """
    if not os.path.exists(image_path):
        print(f"[ERRORE] Immagine non trovata: {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERRORE] Impossibile leggere l'immagine: {image_path}")
        return None

    # Binarizzazione con soglia fissa (le linee della griglia sono scure)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Rilevamento linee orizzontali e verticali con morfologia
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 10, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 10))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    intersections = cv2.bitwise_and(h_lines, v_lines)
    coords = cv2.findNonZero(intersections)

    if coords is None:
        print("[ERRORE] Nessuna griglia rilevata nell'immagine.")
        return None

    coords   = coords[:, 0, :]
    x_coords = cluster_positions([x for x, _ in coords], threshold=10)
    y_coords = cluster_positions([y for _, y in coords], threshold=10)

    detected_cols = len(x_coords) - 1
    detected_rows = len(y_coords) - 1

    if detected_rows <= 0 or detected_cols <= 0:
        print(f"[ERRORE] Griglia non valida: {detected_rows}x{detected_cols}")
        return None

    print(f"[INFO] Griglia rilevata: {detected_rows}x{detected_cols} ({detected_rows * detected_cols} celle)")

    # batch prediction invece di predict cella per cella
    cells_batch = []
    for r in range(detected_rows):
        for c in range(detected_cols):
            x1, x2 = x_coords[c], x_coords[c+1]
            y1, y2 = y_coords[r], y_coords[r+1]
            roi       = img[y1:y2, x1:x2]
            processed = auto_crop_cell(roi)
            cells_batch.append(processed)

    # Stack e normalizzazione in un unico array
    batch_array = np.array(cells_batch, dtype='float32') / 255.0
    batch_array = batch_array[..., np.newaxis]           # shape: (N, 28, 28, 1)

    predictions = model.predict(batch_array, verbose=0)  # shape: (N, 4)
    pred_labels = [CLASS_MAP.get(np.argmax(p), '?') for p in predictions]

    # Verbose: mostra ogni cella con la sua predizione
    if verbose:
        idx = 0
        for r in range(detected_rows):
            for c in range(detected_cols):
                x1, x2 = x_coords[c], x_coords[c+1]
                y1, y2 = y_coords[r], y_coords[r+1]
                roi = img[y1:y2, x1:x2]

                plt.figure(figsize=(4, 2))
                plt.suptitle(f"Cella [{r},{c}]")
                plt.subplot(1, 2, 1); plt.imshow(roi, cmap='gray'); plt.axis('off'); plt.title("Originale")
                plt.subplot(1, 2, 2); plt.imshow(cells_batch[idx], cmap='gray'); plt.axis('off'); plt.title(f"CNN: {pred_labels[idx]}")
                plt.tight_layout(); plt.show()
                idx += 1

    return tuple(pred_labels), detected_rows, detected_cols


# =============================================================================
# LOGICA DI RICERCA - COLORI TERMINAL
# =============================================================================

BLUE  = "\033[34;1m"
RED   = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"


@dataclass
class Result:
    result: Node
    nodes_generated: int
    paths_explored: int
    nodes_left_in_frontier: int


def best_first_graph_search(problem: Problem, f: Callable) -> Result:
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    counter = 1

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return Result(node, counter, len(explored), len(frontier))
        explored.add(node.state)
        for child in node.expand(problem):
            counter += 1
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)

    return Result(None, counter, len(explored), 0)


def ucs(problem: Problem) -> Result:
    return best_first_graph_search(problem, lambda node: node.path_cost)


def greedy(problem: Problem, h: Callable | None = None) -> Result:
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar(problem: Problem, h: Callable | None = None) -> Result:
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda node: h(node) + node.path_cost)


def execute(name: str, algorithm: Callable, problem: Problem, *args, **kwargs) -> None:
    print(f"{RED}{name}{RESET}\n")
    start = time.time()
    sol   = algorithm(problem, *args, **kwargs)
    end   = time.time()

    if problem.goal is not None:
        print(f"\n{GREEN}PROBLEM:{RESET} {problem.initial} -> {problem.goal}")

    if isinstance(sol, Result):
        print(f"{GREEN}Total nodes generated:{RESET}   {sol.nodes_generated}")
        print(f"{GREEN}Paths explored:{RESET}          {sol.paths_explored}")
        print(f"{GREEN}Nodes left in frontier:{RESET}  {sol.nodes_left_in_frontier}")
        sol = sol.result

    print(f"{GREEN}Result:{RESET} {sol.solution() if sol is not None else '---'}")

    if isinstance(sol, Node):
        print(f"{GREEN}Path Cost:{RESET}   {sol.path_cost}")
        print(f"{GREEN}Path Length:{RESET} {sol.depth}")

        print(f"\n{BLUE}=== MATRICE FINALE RISOLTA ==={RESET}")
        final_grid = sol.state.grid
        for r in range(problem.rows):
            print(final_grid[r * problem.cols: (r + 1) * problem.cols])
        print()

    print(f"{GREEN}Time:{RESET} {end - start:.4f} s")
    return sol


# =============================================================================
# DOMINIO: UNIFORM COLORING
# =============================================================================

COSTS = {
    'N': 1, 'S': 1, 'W': 1, 'E': 1,
    'COL-B': 1, 'COL-Y': 2, 'COL-G': 3
}


class ColoringState:
    """
    Stato del dominio Uniform Coloring.
    - grid: tupla con i colori delle celle (la posizione della testina è 'T')
    - held_color: colore "nascosto" sotto la testina
    """

    def __init__(self, grid: tuple, held_color: str, t_pos: int) -> None:
        self.grid = grid
        self.held_color = held_color
        self.t_pos = t_pos #posizione T precalcolata per ottimizzare

    def __hash__(self):
        return hash((self.grid, self.held_color))

    def __eq__(self, other):
        return self.grid == other.grid and self.held_color == other.held_color

    def __lt__(self, other):
        # Necessario per PriorityQueue quando due nodi hanno stesso f(n)
        return self.grid < other.grid


class UniformColoring(Problem):
    def __init__(self, initial_grid: tuple, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.start_pos = initial_grid.index('T')

        total_cells = self.rows * self.cols
        self.X = tuple(i // self.cols for i in range(total_cells))
        self.Y = tuple(i % self.cols for i in range(total_cells))

        initial_state = ColoringState(initial_grid, 'None', self.start_pos)
        super().__init__(initial_state, goal=None)

    def actions(self, state: ColoringState) -> list[str]:
        possible = []
        t = state.t_pos

        # Movimenti cardinali
        if t >= self.cols: possible.append('N')
        if t < len(state.grid) - self.cols: possible.append('S')
        if t % self.cols != 0: possible.append('W')
        if t % self.cols != self.cols - 1: possible.append('E')

        # Colorazioni (solo se non siamo sulla start_pos)
        if t != self.start_pos:
            if state.held_color != 'B': possible.append('COL-B')
            if state.held_color != 'Y': possible.append('COL-Y')
            if state.held_color != 'G': possible.append('COL-G')

        return possible

    def result(self, state: ColoringState, action: str) -> ColoringState:
        t = state.t_pos
        new_grid = list(state.grid)
        new_held = state.held_color
        new_t = t

        if action in ('N', 'S', 'W', 'E'):
            delta = {'N': -self.cols, 'S': self.cols, 'W': -1, 'E': 1}
            neighbor = t + delta[action]

            # Se la T lascia la start_pos, quella cella rimane senza colore ('None')
            # altrimenti ripristina il colore che stava nascondendo
            new_grid[t] = 'None' if t == self.start_pos else state.held_color
            new_held = new_grid[neighbor]  # salva colore della cella destinazione
            new_grid[neighbor] = 'T'
            new_t = neighbor

        elif action.startswith('COL-'):
            new_held = action.split('-')[1]

        return ColoringState(tuple(new_grid), new_held, new_t)

    def path_cost(self, c, state1, action, state2):
        return c + COSTS[action]

    def goal_test(self, state: ColoringState) -> bool:
        """
        Goal:
          1. La testina è tornata alla posizione iniziale (start_pos).
          2. Tutte le celle visibili (≠ 'T') hanno lo stesso colore.
        
        Nota: la start_pos non ha un colore proprio per definizione del dominio,
        quindi held_color = 'None' al ritorno è sempre uno stato valido.
        Non è necessario verificare held_color perché la testina non colora
        mai la propria casella di partenza (vedi actions()).
        """
        if state.t_pos != self.start_pos:
            return False
    
        visible = {c for c in state.grid if c != 'T' and c != 'None'}
        return len(visible) == 1

    def h_combined_cost(self, node: Node) -> int:
        state = node.state
        t = state.t_pos
        s = self.start_pos

        # h_dist
        h_dist = abs(self.X[t] - self.X[s]) + abs(self.Y[t] - self.Y[s])

        # h_paint
        # 1. Prendi tutti i colori reali nella griglia (escludi la testina e la start_pos vuota)
        visible = [c for c in state.grid if c not in ('T', 'None')]
        
        # 2. Aggiungi il colore attualmente coperto dalla testina (se non è vuoto)
        if state.held_color != 'None':
            visible.append(state.held_color)

        if not visible:
            return h_dist

        cost_B = sum(COSTS['COL-B'] for c in visible if c != 'B')
        cost_Y = sum(COSTS['COL-Y'] for c in visible if c != 'Y')
        cost_G = sum(COSTS['COL-G'] for c in visible if c != 'G')
        h_paint = min(cost_B, cost_Y, cost_G)

        return h_dist + h_paint

def generate_gif_simulation(problem, goal_node, output_filename="simulazione_soluzione.gif"):
    """
    Genera una GIF animata che simula il piano di azioni passo dopo passo,
    salvandola nella cartella 'images'.
    """
    if not goal_node:
        print("\n[SIMULATORE] Nessuna soluzione da simulare.")
        return

    print(f"\n[SIMULATORE] Generazione della GIF animata in corso...")
    
    # Crea la directory di output se non esiste
    output_dir = 'gif'
    os.makedirs(output_dir, exist_ok=True)

    # Recupera il percorso completo dei nodi (dallo start al goal)
    path = goal_node.path()
    images = []

    for idx, node in enumerate(path):
        action = node.action
        grid_1d = node.state.grid
        
        # Converte la griglia 1D in una matrice 2D per matplotlib
        table_data = [
            list(grid_1d[r * problem.cols : (r + 1) * problem.cols]) 
            for r in range(problem.rows)
        ]
        
        # Dinamizza la dimensione della figura in base alle colonne/righe
        fig, ax = plt.subplots(figsize=(problem.cols * 1.5, problem.rows * 1.5))
        ax.axis('off')
        
        # Disegna la tabella della griglia
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.scale(1, 2) # Aumenta l'altezza delle celle per renderle quadrate
        table.set_fontsize(20)

        for (row, col), cell in table.get_celld().items():
            if cell.get_text().get_text() == 'T':
                cell.get_text().set_color('red') 
        
        # Aggiungi il nome dell'azione in basso
        label = f"Azione: {action}" if action else "Stato iniziale"
        plt.figtext(0.5, 0.05, label, ha='center', fontsize=16, fontweight='bold')
            
        plt.tight_layout()
        
        # Salva in PNG temporaneo
        fname = os.path.join(output_dir, f'_sim_grid_{idx}.png')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        # Leggi l'immagine temporanea e aggiungila alla lista
        images.append(imageio.imread(fname))
        os.remove(fname) # Pulizia file temporaneo

    # Salva la GIF
    gif_path = os.path.join(output_dir, output_filename)
    imageio.mimsave(gif_path, images, format='GIF', fps=1) # fps=2 per mezzo secondo a frame
    print(f"[SIMULATORE] GIF generata con successo! Salvata in: {gif_path}")


# =============================================================================
# MAIN
# =============================================================================

try:
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Modello '{MODEL_PATH}' caricato con successo.")
except Exception as e:
    print(f"[ERRORE] Caricamento modello: {e}")
    exit()

vision_result = extract_grid_to_matrix(IMAGE_PATH, loaded_model, verbose=False)

if vision_result:
    grid_initial, detected_rows, detected_cols = vision_result

    print(f"\n{BLUE}=== GRIGLIA INIZIALE RILEVATA ({detected_rows}x{detected_cols}) ==={RESET}")
    for r in range(detected_rows):
        print(grid_initial[r * detected_cols: (r + 1) * detected_cols])
    print()

    # FIX: controllo che 'T' sia presente nella griglia prima di creare il problema
    if 'T' not in grid_initial:
        print(f"{RED}[ERRORE] Nessuna 'T' (Testina) rilevata nella griglia. "
              f"Controlla le predizioni CNN o l'immagine.{RESET}")
        exit()

    problem = UniformColoring(grid_initial, detected_rows, detected_cols)

    print("=" * 60)
    risultato_ucs = execute("Uniform Cost Search (UCS)", ucs, problem)
    # Avviamo la simulazione passando il nodo finale
    if risultato_ucs:
        generate_gif_simulation(problem, risultato_ucs, output_filename="simulazione_ucs.gif")

    print("=" * 60)
    risultato_greedy = execute("Greedy Best First Search", greedy, problem, h=problem.h_combined_cost)
    # Avviamo la simulazione passando il nodo finale
    if risultato_greedy:
        generate_gif_simulation(problem, risultato_greedy, output_filename="simulazione_greedy.gif")

    print("=" * 60)
    risultato_astar = execute("A* Search", astar, problem, h=problem.h_combined_cost)
    # Avviamo la simulazione passando il nodo finale
    if risultato_astar:
        generate_gif_simulation(problem, risultato_astar, output_filename="simulazione_astar.gif")

else:
    print(f"{RED}[ERRORE] Analisi immagine fallita o griglia non rilevata.{RESET}")