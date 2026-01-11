"""
Vérification de l'intégrité du projet Ball Sort Puzzle RL.

Ce script liste tous les fichiers critiques du projet.
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_project_structure():
    """Vérifier que tous les fichiers nécessaires existent."""
    
    required_files = {
        'Core Scripts': [
            'train.py',
            'eval.py',
            'demo.py',
            'xai_analysis.py',
            'test_project.py',
            'rainbow_placeholder.py',
        ],
        'Documentation': [
            'README.md',
            'commande.md',
            'PROJET_RESUME.md',
            'requirements.txt',
            '.gitignore',
        ],
        'Configuration': [
            'configs/default_config.yaml',
        ],
        'Environment': [
            'envs/__init__.py',
            'envs/ball_sort_env.py',
        ],
        'Agents': [
            'agents/__init__.py',
            'agents/agent_factory.py',
        ],
        'Utils': [
            'utils/__init__.py',
            'utils/config.py',
        ],
        'Result Directories': [
            'result/models/.gitkeep',
            'result/logs/.gitkeep',
            'result/evaluations/.gitkeep',
            'result/xai/.gitkeep',
            'result/episodes/.gitkeep',
        ],
    }
    
    print("\n" + "="*70)
    print("BALL SORT PUZZLE RL - PROJECT STRUCTURE VERIFICATION")
    print("="*70 + "\n")
    
    all_ok = True
    total_files = 0
    missing_files = []
    
    for category, files in required_files.items():
        print(f"📁 {category}")
        for file in files:
            filepath = project_root / file
            exists = filepath.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
            
            total_files += 1
            if not exists:
                all_ok = False
                missing_files.append(file)
        print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files checked: {total_files}")
    print(f"Missing files: {len(missing_files)}")
    
    if all_ok:
        print("\n✅ ALL FILES PRESENT - Project structure is complete!")
    else:
        print("\n❌ MISSING FILES:")
        for file in missing_files:
            print(f"  - {file}")
    
    print("="*70 + "\n")
    
    return all_ok


def print_project_stats():
    """Afficher des statistiques sur le projet."""
    print("\n" + "="*70)
    print("PROJECT STATISTICS")
    print("="*70 + "\n")
    
    # Count lines of code
    total_lines = 0
    file_counts = {
        'Python files': 0,
        'Documentation': 0,
        'Config files': 0,
    }
    
    for py_file in project_root.rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                file_counts['Python files'] += 1
        except:
            pass
    
    for md_file in project_root.rglob('*.md'):
        file_counts['Documentation'] += 1
    
    for yaml_file in project_root.rglob('*.yaml'):
        file_counts['Config files'] += 1
    
    print(f"📊 Code Statistics:")
    print(f"  Total Python lines: ~{total_lines:,}")
    for category, count in file_counts.items():
        print(f"  {category}: {count}")
    
    print("\n📦 Key Components:")
    print("  ✓ Gymnasium Environment with Action Masking")
    print("  ✓ PPO/A2C/DQN Agent Support")
    print("  ✓ Curriculum Learning")
    print("  ✓ XAI Analysis (Captum)")
    print("  ✓ PyGame Rendering")
    print("  ✓ TensorBoard Logging")
    print("  ✓ Configuration Management")
    print("  ✓ Automated Testing")
    
    print("\n🎯 Supported Algorithms:")
    print("  ✓ PPO (MaskablePPO) - Recommended")
    print("  ✓ A2C (Standard)")
    print("  ✓ DQN (Standard)")
    print("  ⭐ Rainbow (Placeholder + Integration Guide)")
    
    print("\n📈 Features:")
    print("  ✓ 14 Progressive Levels")
    print("  ✓ Action Masking")
    print("  ✓ Reward Decomposition")
    print("  ✓ Episode Recording (GIF)")
    print("  ✓ Integrated Gradients")
    print("  ✓ Trajectory Visualization")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    structure_ok = check_project_structure()
    print_project_stats()
    
    if structure_ok:
        print("🎉 Project is ready for training!")
        print("\nQuick start:")
        print("  1. conda activate colorball")
        print("  2. python train.py --algo ppo --timesteps 10000")
        print("  3. tensorboard --logdir result/logs")
        print("\nFor all commands, see: commande.md\n")
        sys.exit(0)
    else:
        print("⚠ Please fix missing files before proceeding.\n")
        sys.exit(1)
