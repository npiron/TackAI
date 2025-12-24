
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame
import math
import numpy as np
import random

from src.core.constants import W, H, CHECKPOINT_RADIUS, COLOR_ROAD, COLOR_GRASS, COLOR_BORDER

def catmull_rom_spline(P0, P1, P2, P3, n_points=20):
    """
    Calcule n_points sur le segment P1->P2 en utilisant P0 et P3 comme points de contrôle.
    """
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])
    alpha = 0.5
    
    def tj(ti, Pi, Pj):
        return (np.linalg.norm(Pi - Pj))**alpha + ti

    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    t = np.linspace(t1, t2, n_points)
    t = t.reshape(len(t), 1)
    
    A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
    A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
    A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
    
    B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
    B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
    
    C = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
    return list(map(tuple, C)) # Retourne liste de tuples (x,y)

@dataclass
class Track:
    # Propriétés de base
    start_pos: Tuple[float, float]
    start_heading: float
    finish_pos: Tuple[float, float, float] # (x, y, angle) like checkpoints

    # Checkpoints : (x, y, angle_rad) -> Angle remplace 'H'/'V'
    checkpoints: List[Tuple[float, float, float]]
    
    # Masque de collision (Road = 1, Grass = 0)
    collision_mask: pygame.Mask
    
    # Optimisation Numba
    occupancy_grid: np.ndarray # bool 2D array
    
    # Surface visuelle pré-rendue (pour la performance)
    visual_surface: pygame.Surface
    
    # Mathematical representation for alignment reward
    spline_points: np.ndarray # Shape (N, 2)
    
    # Reproducibility
    seed: Optional[int] = None
    
    # Legacy support (optionnel)
    rect_outer: Optional[pygame.Rect] = None
    rect_inner: Optional[pygame.Rect] = None
    walls: Optional[List[pygame.Rect]] = None
    boost_zones: Optional[List[pygame.Rect]] = None

    def get_closest_track_info(self, pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Retourne (tangent_vector, distance_to_center) pour le point de la piste le plus proche.
        Optimisé avec Numpy.
        """
        # 1. Trouver le point le plus proche
        # Broadcasting: pos (2,) - spline_points (N, 2) -> (N, 2)
        diff = self.spline_points - pos
        dists_sq = np.sum(diff * diff, axis=1)
        idx = np.argmin(dists_sq)
        
        # 2. Distance exacte
        dist_to_center = np.sqrt(dists_sq[idx])
        
        # 3. Calculer la tangente à cet endroit
        # On prend le point suivant vs le point actuel (ou précédent)
        # Gestion des indices cycliques
        n_points = len(self.spline_points)
        p_next = self.spline_points[(idx + 1) % n_points]
        p_prev = self.spline_points[(idx - 1 + n_points) % n_points]
        
        # Vecteur tangent centré (plus stable que next - curr)
        tangent = p_next - p_prev
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent /= norm
            
        return tangent, float(dist_to_center)

    def inside_track(self, p: np.ndarray) -> bool:
        """Vérifie si le point est sur la route via le masque."""
        x, y = int(p[0]), int(p[1])
        if 0 <= x < W and 0 <= y < H:
            # Mask.get_at retourne 1 si set, 0 sinon
            return self.collision_mask.get_at((x, y)) == 1
        return False
    
    def on_boost(self, p: np.ndarray) -> bool:
        if not self.boost_zones:
            return False
        x, y = float(p[0]), float(p[1])
        for zone in self.boost_zones:
            if zone.collidepoint(x, y):
                return True
        return False
    
    def hits_wall(self, p: np.ndarray) -> bool:
        """Collision avec les bords de l'écran ou zones interdites."""
        # Dans le mode procédural, si on n'est pas sur la route, c'est un mur (herbe)
        # Mais pour être permissif, disons que hits_wall c'est sortir de l'écran
        x, y = float(p[0]), float(p[1])
        return x < 0 or x > W or y < 0 or y > H

def generate_procedural_track(seed: Optional[int] = None) -> Track:
    """Génère un circuit aléatoire basé sur des splines."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        
    random.seed(seed)
    np.random.seed(seed)
        
    surface = pygame.Surface((W, H))
    surface.fill(COLOR_GRASS)
    
    # 1. Générer points de contrôle pour un VRAI circuit de course
    # On crée un tracé avec des sections variées (lignes droites, virages serrés, courbes rapides)
    center = (W/2, H/2)
    base_radius_x = W * 0.38  # Légèrement plus grand pour avoir de la place
    base_radius_y = H * 0.38
    n_points = 16  # Plus de points = circuit plus long et complexe
    
    points = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        
        # Variation stratégique du rayon pour créer des sections différentes
        # On alterne entre sections rapides (grand rayon) et techniques (petit rayon)
        section_type = i % 4
        
        if section_type == 0:  # Ligne droite / Courbe rapide
            r_var_x = random.uniform(1.1, 1.3)
            r_var_y = random.uniform(1.1, 1.3)
        elif section_type == 1:  # Transition
            r_var_x = random.uniform(0.95, 1.05)
            r_var_y = random.uniform(0.95, 1.05)
        elif section_type == 2:  # Virage serré (épingle)
            r_var_x = random.uniform(0.7, 0.85)
            r_var_y = random.uniform(0.7, 0.85)
        else:  # Courbe moyenne
            r_var_x = random.uniform(0.9, 1.1)
            r_var_y = random.uniform(0.9, 1.1)
        
        px = center[0] + math.cos(angle) * base_radius_x * r_var_x
        py = center[1] + math.sin(angle) * base_radius_y * r_var_y
        
        # Clamp to screen with margins
        px = max(120, min(W-120, px))
        py = max(120, min(H-120, py))
        points.append((px, py))
        
    # Fermer la boucle
    # Pour Catmull-Rom fermé, on a besoin de P-1, P0, P1... Pn, Pn+1
    # On duplique les points pour boucler
    points = points[-1:] + points + points[:2]
    
    spline_points = []
    # Augmenter la densité pour des courbes ultra-lisses
    # Plus de points = meilleure qualité visuelle et physique
    for i in range(len(points) - 3):
        p0, p1, p2, p3 = points[i], points[i+1], points[i+2], points[i+3]
        # 150 points par segment pour des courbes parfaitement fluides
        seg_points = catmull_rom_spline(p0, p1, p2, p3, n_points=150)
        spline_points.extend(seg_points)
        
    # 2. Dessiner la route (Technique du Pinceau Rond)
    # Au lieu de draw.lines qui bug sur les angles, on dessine des disques partout.
    
    # A. La Bordure (Dessous, plus large)
    for p in spline_points:
        pygame.draw.circle(surface, COLOR_BORDER, (int(p[0]), int(p[1])), 70)
        
    # B. La Route (Dessus, normale)
    for p in spline_points:
        pygame.draw.circle(surface, COLOR_ROAD, (int(p[0]), int(p[1])), 60)
    
    # Create mask surface for occupancy grid generation
    mask_surf = pygame.Surface((W, H))
    mask_surf.fill((0, 0, 0))
    for p in spline_points:
        pygame.draw.circle(mask_surf, (255, 255, 255), (int(p[0]), int(p[1])), 60)
    
    # Random obstacles removed for clean racing

    # Convertir en mask pygame
    collision_mask = pygame.mask.from_threshold(mask_surf, (255, 255, 255), (10, 10, 10))
    
    # === OPTIMISATION NUMBA ===
    # Convertir en grille Numpy (C-contiguous, bool) pour le JIT
    # surfarray est (W, H), on veut (H, W) pour l'accès [row, col] -> [y, x]
    # Ou on garde (W, H) et on accède [x, y]. JIT ops utilise [py, px]. 
    # Mon jit_ops utilise grid[py, px], donc il attend (H, W).
    # Pygame surfarray.array3d retourne (Width, Height, 3).
    # Donc arr[x, y].
    # Je vais transpose pour avoir (Height, Width) -> arr[y, x].
    arr = pygame.surfarray.array3d(mask_surf)
    # Blanc = (255,255,255) => Route. Noir = (0,0,0) => Mur.
    # On prend canal rouge > 128.
    occupancy_grid = (arr[:, :, 0] > 128).T.astype(np.bool_)
     
    # 3. Placer Checkpoints
    
    # 3. Placer Checkpoints
    # On en place régulièremet sur la spline
    checkpoints = []
    step = len(spline_points) // 8 # 8 Checkpoints
    
    for i in range(0, len(spline_points), step):
        if len(checkpoints) >= 8: break
        
        p = spline_points[i]
        
        # Calcul de la tangente pour l'orientation
        # Next point
        next_p = spline_points[(i + 5) % len(spline_points)]
        dx = next_p[0] - p[0]
        dy = next_p[1] - p[1]
        angle = math.atan2(dy, dx)
        
        checkpoints.append((p[0], p[1], angle + math.pi/2)) # checkpoint perpendiculaire route
        
    start_idx = 0
    start_pos = spline_points[start_idx]
    
    # Heading initial: vers le point suivant
    p_next = spline_points[5]
    start_heading = math.atan2(p_next[1]-start_pos[1], p_next[0]-start_pos[0])
    
    # Finish Line
    f_idx = -10
    p_finish = spline_points[f_idx]
    
    # Calculate orientation for finish line
    # Depending on list length, f_idx+5 might wrap or be valid. 
    # spline_points is a list. Negative index works if valid range.
    # To be safe, we use modulo logic relative to length.
    L = len(spline_points)
    idx_next = (L + f_idx + 5) % L
    
    p_finish_next = spline_points[idx_next]
    
    dx_f = p_finish_next[0] - p_finish[0]
    dy_f = p_finish_next[1] - p_finish[1]
    finish_angle = math.atan2(dy_f, dx_f) + math.pi/2
    
    finish_pos = (p_finish[0], p_finish[1], finish_angle)
    
    # Boost zones? On verra plus tard
    
    return Track(
        start_pos=start_pos,
        start_heading=start_heading,
        finish_pos=finish_pos,
        checkpoints=checkpoints,
        collision_mask=collision_mask,
        occupancy_grid=occupancy_grid,
        visual_surface=surface,
        spline_points=np.array(spline_points), # Store raw points for alignment logic
        seed=seed, # Store the seed used for generation
        walls=[], # Pas de murs physiques explicites, le hors-piste est géré par mask
        boost_zones=[]
    )

def build_track(seed: int = 42) -> Track:
    """Wrapper pour conserver la compatibilité, mais appelle le procedural par défaut."""
    # Note: On utilise une seed fixe ici pour que 'manual play' soit constant au début
    # Mais le training pourra appeler generate_procedural_track(seed=None)
    return generate_procedural_track(seed=seed)
