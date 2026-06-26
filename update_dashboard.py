"""
Update Dashboard for Multi-Plant Support
=========================================
Automatically updates the dashboard to show all plant species
and diseases from the PlantVillage dataset.
"""

import os


def update_dashboard_with_plantvillage():
    """Update frontend/app.py to support PlantVillage disease labels."""
    
    # Check if mapping file exists
    if not os.path.exists('plant_disease_mapping.txt'):
        print("❌ Error: plant_disease_mapping.txt not found!")
        print("\nYou need to run download_plantvillage.py first to prepare the data.")
        return
    
    # Read the disease mapping
    disease_labels = []
    with open('plant_disease_mapping.txt', 'r') as f:
        lines = f.readlines()[2:]  # Skip header
        for line in lines:
            if ':' in line:
                parts = line.strip().split(':')
                label_num = int(parts[0].strip())
                label_text = parts[1].strip()
                disease_labels.append((label_num, label_text))
    
    print(f"\n✓ Loaded {len(disease_labels)} disease classes")
    
    # Create Python list for insertion
    disease_list_str = "[\n"
    for label_num, label_text in sorted(disease_labels):
        disease_list_str += f'        "{label_text}",  # {label_num}\n'
    disease_list_str += "    ]"
    
    print("\n" + "="*70)
    print("Dashboard Update Code")
    print("="*70)
    print("\nAdd this to frontend/app.py (replace the disease_db dict):\n")
    
    code = f"""
# Multi-plant disease database (PlantVillage)
PLANTVILLAGE_DISEASES = {disease_list_str}

def display_predictions(predictions, weather_data):
    \"\"\"Display multi-task prediction results.\"\"\"
    
    # Use PlantVillage disease labels
    disease_labels = PLANTVILLAGE_DISEASES
    
    cycle_labels = ["Spring", "Summer", "Fall", "Winter"]
    resource_labels = ["Water", "Nitrogen", "Phosphorus", "Potassium"]
"""
    
    print(code)
    
    print("\n" + "="*70)
    print("Automatic Update Option")
    print("="*70)
    
    auto_update = input("\nWould you like me to automatically update frontend/app.py? (y/n): ").strip().lower()
    
    if auto_update == 'y':
        # Backup original
        import shutil
        shutil.copy('frontend/app.py', 'frontend/app.py.backup')
        print("\n✓ Created backup: frontend/app.py.backup")
        
        # Read current app.py
        with open('frontend/app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace the display_predictions function
        import re
        
        # Add disease list at the top after imports
        insert_point = content.find('# Page configuration')
        if insert_point == -1:
            insert_point = content.find('st.set_page_config')
        
        new_disease_section = f"\n# Multi-plant disease database (PlantVillage)\nPLANTVILLAGE_DISEASES = {disease_list_str}\n\n"
        
        content = content[:insert_point] + new_disease_section + content[insert_point:]
        
        # Replace disease_db section in display_predictions
        pattern = r"disease_db = \{[^}]+\}"
        replacement = "# Using PlantVillage diseases\n    disease_labels = PLANTVILLAGE_DISEASES"
        
        # First, try to replace the disease_db dict
        if 'disease_db' in content:
            # Find the start of disease_db
            start = content.find('disease_db = {')
            if start != -1:
                # Find the matching closing brace
                brace_count = 0
                i = start + len('disease_db = ')
                while i < len(content):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the end
                            end = i + 1
                            # Also remove the following line that uses disease_db.get
                            next_line_end = content.find('\n', end) + 1
                            if 'disease_labels = disease_db.get' in content[end:next_line_end+100]:
                                end = content.find('\n', end + 50) + 1
                            
                            content = content[:start] + "# Using PlantVillage diseases\n    disease_labels = PLANTVILLAGE_DISEASES\n" + content[end:]
                            break
                    i += 1
        
        # Remove crop type selection from sidebar (not needed for multi-plant)
        crop_section_start = content.find('# Crop selection')
        if crop_section_start != -1:
            crop_section_end = content.find('st.session_state[\'crop_type\'] = crop_type', crop_section_start)
            if crop_section_end != -1:
                crop_section_end = content.find('\n', crop_section_end) + 1
                content = content[:crop_section_start] + content[crop_section_end:]
        
        # Write updated content
        with open('frontend/app.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✓ Updated frontend/app.py")
        print("\n✅ Dashboard updated successfully!")
        print("\nThe dashboard now supports all PlantVillage diseases.")
    else:
        print("\n📝 Manual update instructions:")
        print("  1. Open frontend/app.py")
        print("  2. Add the PLANTVILLAGE_DISEASES list after imports")
        print("  3. Replace disease_db with: disease_labels = PLANTVILLAGE_DISEASES")
        print("  4. Remove crop type selection (optional)")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Dashboard Updater for PlantVillage Dataset")
    print("="*70)
    
    update_dashboard_with_plantvillage()
    
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("\n1. ✓ Dashboard updated")
    print("2. ⏳ Update core/train.py:")
    print(f"     Change 'num_diseases=10' to match your dataset")
    print("3. 🚀 Start training:")
    print("     python core/train.py")


if __name__ == "__main__":
    main()
