README - Stratégie de Trading de Paires (Pair Trading)
Ce projet implémente une stratégie de trading de paires (Pair Trading) basée sur une approche statistique. L'objectif est d'identifier des opportunités d'arbitrage entre deux actifs financiers corrélés en exploitant les écarts temporaires de leurs prix. La stratégie utilise des outils statistiques tels que la normalisation des prix, la mesure de distance, la régression linéaire, et le test de Dickey-Fuller pour déterminer les conditions d'entrée et de sortie.

Fonctionnalités Principales
Normalisation des Prix :

Les prix des actifs sont normalisés pour permettre une comparaison directe et une analyse cohérente.

Mesure de Distance :

Une mesure de distance est calculée pour quantifier l'écart entre les prix normalisés des deux actifs.

Régression Linéaire :

Un coefficient de régression (beta) est estimé pour modéliser la relation entre les deux actifs.

Test de Dickey-Fuller :

Le test de Dickey-Fuller est utilisé pour vérifier la stationnarité des résidus, ce qui est essentiel pour valider la stratégie.

Conditions d'Entrée et de Sortie :

Entrée : Lorsque l'écart entre les prix dépasse un seuil dynamique, une position est ouverte.

Sortie : La position est fermée lorsque l'écart revient à un niveau normal ou lorsque les conditions de stop-loss/take-profit sont atteintes.

Gestion des Risques :

La stratégie intègre des outils de gestion des risques, tels que le calcul de la Value at Risk (VaR) et le suivi du drawdown.

Structure du Code
Classes Principales
PairTradingConfig :

Contient les paramètres de configuration de la stratégie, tels que les périodes de formation, les seuils, et les actifs cibles.

PairTradingCalculus :

Stocke les calculs intermédiaires de la stratégie, y compris les prix normalisés, les mesures de distance, les résidus, et les résultats du test de Dickey-Fuller.

StrategySignals :

Contient les signaux d'entrée et de sortie générés par la stratégie.

PairTradingBackTest :

Stocke les résultats du backtest, y compris les changements de prix, les poids du portefeuille, les rendements, et le ratio de Sharpe.

PairTradingCost :

Gère les coûts de transaction, y compris les frais fixes, le slippage, et les coûts liés au spread bid-ask.

PairTradingRisk :

Contient les outils de gestion des risques, tels que le calcul de la VaR et du drawdown.

PairTradingHistory :

Agrége tous les éléments précédents pour former un historique complet de la stratégie.

PairTradingBacktestStrategy :

Implémente la logique de la stratégie, y compris les calculs intermédiaires, les conditions d'entrée/sortie, et la gestion des risques.

Utilisation
Configuration
Paramètres de la Stratégie :

formation_period : Période de formation pour les calculs statistiques.

threshold_parameter : Paramètre de seuil pour les conditions d'entrée.

underlying : Actifs cibles pour la stratégie.

Exemple de Configuration :

python
Copy
config = PairTradingConfig(
    formation_period=60,
    threshold_parameter=2.0,
    underlying={"asset1": "AAPL", "asset2": "MSFT"}
)
Exécution de la Stratégie
Initialisation :

python
Copy
strategy = PairTradingBacktestStrategy(config)
Calcul des Composants Statistiques :

Utilisez les méthodes calc_normalized_prices, calc_distance_measure, calc_beta, et calc_resid pour calculer les composants statistiques.

Vérification des Conditions d'Entrée/Sortie :

Utilisez criteria_decision pour déterminer les opportunités d'entrée.

Utilisez calc_threshold pour ajuster les seuils dynamiques.

Gestion des Positions :

Les positions sont gérées automatiquement en fonction des conditions d'entrée/sortie et des niveaux de stop-loss/take-profit.

Exemple de Workflow
python
Copy
# Initialisation
strategy = PairTradingBacktestStrategy(config)

# Simulation sur une période donnée
start_date = dt.date(2023, 1, 1)
end_date = dt.date(2023, 12, 31)
current_date = start_date

while current_date <= end_date:
    # Calcul des composants statistiques
    normalized_prices = strategy.calc_normalized_prices("AAPL", current_date)
    distance_measure = strategy.calc_distance_measure("AAPL", "MSFT", current_date)
    beta = strategy.calc_beta("AAPL", "MSFT", current_date)
    resid = strategy.calc_resid("AAPL", "MSFT", current_date)

    # Vérification des conditions d'entrée
    if strategy.criteria_decision("AAPL", "MSFT", current_date):
        strategy.open_position("AAPL", "MSFT", current_date)
        logger.info(f"Entry signal on {current_date}")

    # Vérification des conditions de sortie
    if strategy.check_exit_condition("AAPL", "MSFT", current_date):
        strategy.close_position("AAPL", "MSFT", current_date)
        logger.info(f"Exit signal on {current_date}")

    # Passage au jour suivant
    current_date = strategy.calendar.busday_add(current_date, 1)
Dépendances
numpy : Pour les calculs numériques.

statsmodels : Pour le test de Dickey-Fuller.

scipy : Pour les calculs statistiques avancés.

pydantic : Pour la validation des configurations.

loguru : Pour la journalisation des événements.

grt_lib_price_loader : Pour le chargement des données de prix.

grt_lib_orchestrator : Pour l'orchestration des stratégies de backtest.

grt_lib_order_book : Pour la gestion des ordres et des transactions.

Améliorations Possibles
Optimisation des Paramètres :

Utiliser des techniques d'optimisation pour trouver les meilleures périodes et seuils pour la stratégie.

Backtesting :

Implémenter un backtest complet pour évaluer la performance de la stratégie sur des données historiques.

Gestion des Risques Avancée :

Ajouter des fonctionnalités de gestion des risques, telles que la diversification du portefeuille et la gestion des corrélations entre actifs.

Visualisation :

Ajouter des graphiques pour visualiser les niveaux de prix, les signaux d'entrée/sortie, et la performance de la stratégie.

Conclusion
Cette stratégie de trading de paires combine des techniques statistiques avancées avec une gestion de risque robuste pour identifier des opportunités d'arbitrage sur les marchés financiers. Elle est conçue pour être flexible et adaptable
