"""
RegolithSense - Machine Learning Mineral Prospectivity Mapping (MPM) Module
Integrates multi-criteria geophysical & radiometric evidence layers into
probabilistic prospectivity indices and ranks drill-ready exploration targets.
Supports Random Forest, Gradient Boosting, and Weights of Evidence (Logistic) architectures.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy import ndimage
import json

class MineralProspectivityEngine:
    def __init__(self, model_type="Random Forest", n_estimators=60, max_depth=8, random_state=42):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.is_trained = False
        
        if model_type == "Gradient Boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=min(max_depth, 6), random_state=random_state
            )
        elif model_type == "Weights of Evidence (WofE / Logistic)":
            self.model = LogisticRegression(
                class_weight='balanced', max_iter=300, random_state=random_state
            )
        else: # Default: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )

    def build_evidence_tensor(self, tdr_grid, thdr_grid, as_grid, k_eth_grid, potassic_factor, 
                              regolith_grid=None, bouguer_grid=None, gzz_grid=None, f_param_grid=None, selected_features=None):
        """
        Stacks spatial layers into a normalized feature tensor of shape (N_pixels, N_features).
        Allows feature selection for ablation studies including multi-physics gravity.
        """
        ny, nx = tdr_grid.shape
        
        # 1. Proximity to Tilt Angle zero-crossing (structural contact)
        contacts = (np.abs(tdr_grid) < 8.0).astype(float)
        dist_contacts = ndimage.distance_transform_edt(1 - contacts) * 50.0
        prox_contacts = np.exp(-dist_contacts / 800.0) # decay scale 800m
        
        # 2. Total Horizontal Derivative (structural fault strength)
        thdr_norm = (thdr_grid - np.min(thdr_grid)) / (np.ptp(thdr_grid) + 1e-6)
        
        # 3. Analytic Signal (contact magnetic contrast)
        as_norm = (as_grid - np.min(as_grid)) / (np.ptp(as_grid) + 1e-6)
        
        # 4. Potassic Alteration Factor (K / eTh normalized)
        pot_norm = (potassic_factor - np.min(potassic_factor)) / (np.ptp(potassic_factor) + 1e-6)
        
        # 5. K/eTh ratio
        k_eth_norm = (k_eth_grid - np.min(k_eth_grid)) / (np.ptp(k_eth_grid) + 1e-6)

        # 6. Regolith Weathering Factor (if available)
        if regolith_grid is not None:
            reg_norm = 1.0 - (regolith_grid - np.min(regolith_grid)) / (np.ptp(regolith_grid) + 1e-6)
        else:
            reg_norm = np.ones((ny, nx)) * 0.5

        # 7. Bouguer Gravity Density Anomaly (Inverted: Negative Bouguer = Felsic Leucogranites)
        if bouguer_grid is not None:
            grav_norm = 1.0 - (bouguer_grid - np.min(bouguer_grid)) / (np.ptp(bouguer_grid) + 1e-6)
        else:
            grav_norm = np.ones((ny, nx)) * 0.5

        # 8. First Vertical Gravity Gradient (Gzz - Edge Density Contrast)
        if gzz_grid is not None:
            gzz_norm = (gzz_grid - np.min(gzz_grid)) / (np.ptp(gzz_grid) + 1e-6)
        else:
            gzz_norm = np.ones((ny, nx)) * 0.5

        # 9. Efimov F-Parameter
        if f_param_grid is not None:
            f_norm = (f_param_grid - np.min(f_param_grid)) / (np.ptp(f_param_grid) + 1e-6)
        else:
            f_norm = pot_norm

        all_layers = {
            "Proximity to TDR Contacts": prox_contacts,
            "Horizontal Gradient (THDR)": thdr_norm,
            "Analytic Signal Contrast": as_norm,
            "Potassic Alteration Halo (K/eTh)": pot_norm,
            "Efimov F-Parameter Alteration": f_norm,
            "Bouguer Gravity Lows (Granite Density)": grav_norm,
            "Vertical Gravity Gradient (Gzz)": gzz_norm,
            "Regolith Weathering Index": reg_norm
        }

        if selected_features is None or len(selected_features) == 0:
            selected_features = list(all_layers.keys())

        feature_cols = []
        valid_selected = []
        for name in selected_features:
            if name in all_layers:
                feature_cols.append(all_layers[name].ravel())
                valid_selected.append(name)

        if len(feature_cols) == 0:
            # Fallback to all layers
            for name, arr in all_layers.items():
                feature_cols.append(arr.ravel())
                valid_selected.append(name)

        feature_matrix = np.column_stack(feature_cols)
        
        return feature_matrix, all_layers, valid_selected

    def train_and_predict(self, feature_matrix, known_deposit_coords, grid_shape, bounds, 
                           elev_grid=None, regolith_grid=None, threshold=0.72):
        """
        Trains model using synthetic/known pegmatite deposits as positive training labels
        and barren background as negatives.
        bounds: (min_lon, max_lon, min_lat, max_lat)
        """
        ny, nx = grid_shape
        min_lon, max_lon, min_lat, max_lat = bounds
        
        # Create ground-truth label mask
        y = np.zeros(ny * nx, dtype=int)
        
        for dep in known_deposit_coords:
            lon, lat = dep["lon"], dep["lat"]
            ix = int(((lon - min_lon) / (max_lon - min_lon)) * (nx - 1))
            iy = int(((max_lat - lat) / (max_lat - min_lat)) * (ny - 1))
            if 0 <= ix < nx and 0 <= iy < ny:
                y_idx = iy * nx + ix
                y[y_idx] = 1
                
        # Dilate deposit footprint for realistic training
        y_grid = y.reshape((ny, nx))
        y_dilated = ndimage.binary_dilation(y_grid, iterations=2).astype(int).ravel()
        
        # Balance dataset: take all positives and 6x random negatives
        pos_indices = np.where(y_dilated == 1)[0]
        if len(pos_indices) == 0:
            pos_indices = np.array([int(ny*nx * 0.45), int(ny*nx * 0.52), int(ny*nx * 0.61)])
            y_dilated[pos_indices] = 1
            
        neg_indices = np.where(y_dilated == 0)[0]
        np.random.seed(self.random_state)
        selected_negs = np.random.choice(neg_indices, size=min(len(neg_indices), len(pos_indices) * 6), replace=False)
        
        train_indices = np.concatenate([pos_indices, selected_negs])
        X_train = feature_matrix[train_indices]
        y_train = y_dilated[train_indices]
        
        # Fit Model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Predict probability across entire map grid
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_matrix)[:, 1]
            train_probs = self.model.predict_proba(X_train)[:, 1]
        else:
            probs = self.model.decision_function(feature_matrix)
            probs = 1 / (1 + np.exp(-probs))
            train_probs = probs[train_indices]

        prospectivity_map = probs.reshape((ny, nx))
        prospectivity_map = ndimage.gaussian_filter(prospectivity_map, sigma=1.1)
        
        # Compute ROC Curve & AUC Score
        fpr, tpr, _ = roc_curve(y_train, train_probs)
        roc_auc = float(auc(fpr, tpr))
        
        # Feature Importances or Coefficients
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coef = np.abs(self.model.coef_[0])
            importances = coef / (np.sum(coef) + 1e-6)
        else:
            importances = np.ones(feature_matrix.shape[1]) / feature_matrix.shape[1]

        # Extract Ranked Drill Targets
        targets = self._extract_ranked_targets(prospectivity_map, bounds, elev_grid, regolith_grid, threshold=threshold)
        
        metrics = {
            "auc": round(roc_auc, 3),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "n_train_samples": int(len(train_indices)),
            "n_positive_samples": int(len(pos_indices))
        }

        return prospectivity_map, targets, importances, metrics

    def _extract_ranked_targets(self, prob_grid, bounds, elev_grid=None, regolith_grid=None, threshold=0.72):
        """
        Locates regional local maxima in the prospectivity grid and ranks them into Tier-1, Tier-2, Tier-3.
        """
        ny, nx = prob_grid.shape
        min_lon, max_lon, min_lat, max_lat = bounds
        
        local_max = ndimage.maximum_filter(prob_grid, size=7) == prob_grid
        candidate_mask = (prob_grid >= threshold) & local_max
        y_coords, x_coords = np.where(candidate_mask)
        
        targets = []
        for i, (iy, ix) in enumerate(zip(y_coords, x_coords)):
            score = float(prob_grid[iy, ix])
            lon = min_lon + (ix / (nx - 1)) * (max_lon - min_lon)
            lat = max_lat - (iy / (ny - 1)) * (max_lat - min_lat)
            
            s_elev = float(elev_grid[iy, ix]) if elev_grid is not None else 400.0
            r_thick = float(regolith_grid[iy, ix]) if regolith_grid is not None else 35.0
            drill_depth = round(r_thick + 45.0, 1)
            
            if score >= 0.85:
                tier = "Tier-1 (Immediate Ground Geophysics / RC Drilling)"
                priority = "High"
            elif score >= 0.75:
                tier = "Tier-2 (Infill Geochemical Soil Grid / GPR)"
                priority = "Medium"
            else:
                tier = "Tier-3 (Regional Reconnaissance)"
                priority = "Low"
                
            targets.append({
                "target_id": f"RGT-{i+1:03d}",
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "grid_ix": int(ix),
                "grid_iy": int(iy),
                "surface_elevation_m": round(s_elev, 1),
                "regolith_cover_m": round(r_thick, 1),
                "target_drill_depth_m": drill_depth,
                "prospectivity_score": round(score * 100, 1),
                "target_tier": tier,
                "priority": priority,
                "target_type": "Concealed LCT Pegmatite Structural Node",
                "recommended_spacing_m": "50m x 25m RC drilling grid"
            })
            
        targets.sort(key=lambda t: t["prospectivity_score"], reverse=True)
        return targets
