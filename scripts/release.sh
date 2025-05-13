#!/bin/bash

set -e  # Exit on error

# Get latest tag
latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")
echo "Latest tag: $latest_tag"

# Strip 'v' and split version
version=${latest_tag#v}
IFS='.' read -r major minor patch <<< "$version"

# Increment patch (default behavior)
patch=$((patch + 1))

# Compose new version
new_version="v$major.$minor.$patch"

# Confirm or allow override via CLI
if [ "$1" == "minor" ]; then
    minor=$((minor + 1))
    patch=0
    new_version="v$major.$minor.$patch"
elif [ "$1" == "major" ]; then
    major=$((major + 1))
    minor=0
    patch=0
    new_version="v$major.$minor.$patch"
fi

echo "Releasing new version: $new_version"

# Create annotated tag
git tag -a "$new_version" -m "Release $new_version"

# Push tag to remote
git push origin "$new_version"

# Create a release
gh release create "$new_version" \
    --title "$new_version" \
    --notes "Release $new_version" \
    --target "$latest_tag" \
    --generate-notes

# Push the release to GitHub
gh release upload "$new_version" \
    --clobber \
    --title "$new_version" \
    --notes "Release $new_version" \
    --target "$latest_tag" \
    --generate-notes

echo "✅ Version $new_version released and pushed!"
