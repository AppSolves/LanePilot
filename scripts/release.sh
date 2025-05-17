#!/bin/bash

set -e  # Exit on error

# Get latest tag
latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")
echo "Latest tag: $latest_tag"

# Strip 'v' and split version
version=${latest_tag#v}
IFS='.' read -r major minor patch <<< "$version"

# Default: increment patch
if [ $# -eq 0 ] || [ "$1" == "patch" ]; then
    minor=$((minor))
    patch=$((patch + 1))
    new_version="v$major.$minor.$patch"
    release_type="patch"
elif [ "$1" == "minor" ]; then
    minor=$((minor + 1))
    patch=0
    new_version="v$major.$minor.$patch"
    release_type="minor"
elif [ "$1" == "major" ]; then
    major=$((major + 1))
    minor=0
    patch=0
    new_version="v$major.$minor.$patch"
    release_type="major"
else
    echo "❌ Error: Invalid argument '$1'. Use no argument (default: 'patch'), 'minor', or 'major'."
    exit 1
fi

echo "Releasing new $release_type version: $new_version"

# Create annotated tag
git tag -a "$new_version" -m "Release $new_version (automated $release_type update)"

# Push tag to remote
git push origin "$new_version"

# Define multiline release notes
release_notes=$(cat <<EOF
Release $new_version (automated $release_type update)

### Update Docker Images: ###
> [!NOTE]
> Pull the latest Docker images for the newest features and bug fixes.
> Use the following commands to pull the latest images:

- Raspberry Pi:
\`\`\`bash
curl -sSL https://raw.githubusercontent.com/AppSolves/LanePilot/refs/heads/main/scripts/compose.sh | bash -s raspberry
\`\`\`

- NVIDIA Jetson:
\`\`\`bash
curl -sSL https://raw.githubusercontent.com/AppSolves/LanePilot/refs/heads/main/scripts/compose.sh | bash -s jetson
\`\`\`

> [!TIP]
> It is strongly recommended to enable [SUPER MAXN mode](https://www.jetson-ai-lab.com/initial_setup_jon.html#8-unlock-super-performance) on your Jetson device.
> This mode allows the Jetson to run at maximum performance, which is beneficial for AI workloads.
EOF
)

# Create a release
gh release create "$new_version" \
    --title "$new_version" \
    --notes "$release_notes" \
    --generate-notes

echo "✅ Version $new_version released and pushed!"