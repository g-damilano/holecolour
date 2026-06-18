param(
    [string]$ProjectRoot = "",
    [string]$ConfigPath = "",
    [switch]$DryRun,
    [switch]$NoAutoDetect
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-ConfigValue {
    param(
        $Config,
        [Parameter(Mandatory = $true)][string]$Name,
        $Default = $null
    )

    if ($null -eq $Config) {
        return $Default
    }

    $property = $Config.PSObject.Properties | Where-Object { $_.Name -eq $Name } | Select-Object -First 1
    if ($null -eq $property) {
        return $Default
    }

    return $property.Value
}

function Test-AutoValue {
    param($Value)

    if ($null -eq $Value) {
        return $true
    }

    if ($Value -is [System.Array]) {
        if ($Value.Count -eq 0) {
            return $true
        }
        if ($Value.Count -eq 1) {
            return Test-AutoValue $Value[0]
        }
        return $false
    }

    $text = ([string]$Value).Trim().ToLowerInvariant()
    return [string]::IsNullOrWhiteSpace($text) -or $text -in @("auto", "detect", "infer")
}

function Convert-ToStringArray {
    param($Value)

    if ($null -eq $Value) {
        return @()
    }

    return @($Value) | ForEach-Object { ([string]$_).Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

function Add-UniqueValue {
    param(
        [System.Collections.Generic.List[string]]$List,
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }

    $normalized = (Convert-ToZipPath $Value).Trim("/")
    if (-not $List.Contains($normalized)) {
        $List.Add($normalized) | Out-Null
    }
}

function Convert-ToRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $rootUri = [System.Uri](([System.IO.Path]::GetFullPath($Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar))
    $pathUri = [System.Uri]([System.IO.Path]::GetFullPath($Path))
    return [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString()).Replace("/", [System.IO.Path]::DirectorySeparatorChar)
}

function Convert-ToZipPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace([string][System.IO.Path]::DirectorySeparatorChar, "/").Replace([string][System.IO.Path]::AltDirectorySeparatorChar, "/")
}

function Resolve-UnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $rootFull = Get-FullPath $Root
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        Get-FullPath $Path
    } else {
        Get-FullPath (Join-Path $rootFull $Path)
    }

    $rootPrefix = $rootFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if ($candidate -ne $rootFull -and -not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside project root: $Path"
    }

    return $candidate
}

function Test-RepositoryMarker {
    param([Parameter(Mandatory = $true)][string]$Path)

    foreach ($marker in @(".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "settings.gradle")) {
        if (Test-Path -LiteralPath (Join-Path $Path $marker)) {
            return $true
        }
    }

    if (Get-ChildItem -LiteralPath $Path -Filter "*.sln" -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
        return $true
    }

    return $false
}

function Find-RepositoryRoot {
    param([Parameter(Mandatory = $true)][string]$StartPath)

    $candidate = Get-FullPath $StartPath
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $candidate = Split-Path -Parent $candidate
    }

    while ($true) {
        if (Test-RepositoryMarker $candidate) {
            return $candidate
        }

        $parent = Split-Path -Parent $candidate
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $candidate) {
            return (Get-FullPath $StartPath)
        }
        $candidate = $parent
    }
}

function Get-TomlSection {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $escaped = [regex]::Escape($Name)
    $match = [regex]::Match($Text, "(?ms)^\[$escaped\]\s*(.*?)(?=^\[|\z)")
    if ($match.Success) {
        return $match.Groups[1].Value
    }

    return ""
}

function Get-TomlString {
    param(
        [Parameter(Mandatory = $true)][string]$Section,
        [Parameter(Mandatory = $true)][string]$Key
    )

    $escaped = [regex]::Escape($Key)
    $match = [regex]::Match($Section, "(?m)^\s*$escaped\s*=\s*[""']([^""']+)[""']")
    if ($match.Success) {
        return $match.Groups[1].Value
    }

    return $null
}

function Get-TomlArrayStrings {
    param(
        [Parameter(Mandatory = $true)][string]$Section,
        [Parameter(Mandatory = $true)][string]$Key
    )

    $values = [System.Collections.Generic.List[string]]::new()
    $escaped = [regex]::Escape($Key)
    $match = [regex]::Match($Section, "(?ms)^\s*$escaped\s*=\s*\[(.*?)\]")
    if (-not $match.Success) {
        return @()
    }

    foreach ($item in [regex]::Matches($match.Groups[1].Value, "[""']([^""']+)[""']")) {
        Add-UniqueValue -List $values -Value $item.Groups[1].Value
    }

    return @($values)
}

function Add-ExistingPath {
    param(
        [System.Collections.Generic.List[string]]$List,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $fullPath = Join-Path $Root $RelativePath
    if (Test-Path -LiteralPath $fullPath) {
        Add-UniqueValue -List $List -Value $RelativePath
    }
}

function Add-ExistingPaths {
    param(
        [System.Collections.Generic.List[string]]$List,
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$RelativePaths
    )

    foreach ($relativePath in $RelativePaths) {
        Add-ExistingPath -List $List -Root $Root -RelativePath $relativePath
    }
}

function Get-PythonPackagePaths {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PyProjectText
    )

    $paths = [System.Collections.Generic.List[string]]::new()
    $setuptoolsSection = Get-TomlSection -Text $PyProjectText -Name "tool.setuptools"
    $findSection = Get-TomlSection -Text $PyProjectText -Name "tool.setuptools.packages.find"
    $packageDir = "."

    $packageDirMatch = [regex]::Match($setuptoolsSection, "(?m)^\s*package-dir\s*=\s*\{[^\r\n]*[""']{0,2}[""']\s*=\s*[""']([^""']+)[""']")
    if ($packageDirMatch.Success) {
        $packageDir = $packageDirMatch.Groups[1].Value
    }

    $includePatterns = Get-TomlArrayStrings -Section $findSection -Key "include"
    foreach ($pattern in $includePatterns) {
        $base = $pattern -replace "\.\*$", "" -replace "\*$", ""
        $base = $base.Trim(".")
        if ([string]::IsNullOrWhiteSpace($base)) {
            continue
        }

        $relative = $base.Replace(".", [string][System.IO.Path]::DirectorySeparatorChar)
        if ($packageDir -ne ".") {
            $relative = Join-Path $packageDir $relative
        }
        Add-ExistingPath -List $paths -Root $Root -RelativePath $relative
    }

    if ($paths.Count -gt 0) {
        return @($paths)
    }

    $searchRelative = if ($packageDir -eq ".") { "." } else { $packageDir }
    $searchRoot = Join-Path $Root $searchRelative
    if (Test-Path -LiteralPath $searchRoot) {
        $packageDirs = Get-ChildItem -LiteralPath $searchRoot -Directory -Force |
            Where-Object {
                -not $_.Name.StartsWith(".") -and
                $_.Name -notin @("__pycache__", "build", "dist", "node_modules") -and
                (Test-Path -LiteralPath (Join-Path $_.FullName "__init__.py"))
            }

        foreach ($package in $packageDirs) {
            $relative = Convert-ToRelativePath -Root $Root -Path $package.FullName
            Add-UniqueValue -List $paths -Value $relative
        }
    }

    if ($paths.Count -eq 0) {
        Add-ExistingPath -List $paths -Root $Root -RelativePath "src"
    }

    return @($paths)
}

function Get-AutoInitialInputPaths {
    param([Parameter(Mandatory = $true)][string]$Root)

    $paths = [System.Collections.Generic.List[string]]::new()
    $extensions = @(".mp4", ".avi", ".mov", ".mkv", ".csv", ".json", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".wav")
    $namePattern = "^(input|sample|example|fixture|seed)([_.-].*)?$"

    Get-ChildItem -LiteralPath $Root -File -Force -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Where-Object { $_.BaseName -match $namePattern -and $_.Extension.ToLowerInvariant() -in $extensions } |
        ForEach-Object {
            $relative = Convert-ToRelativePath -Root $Root -Path $_.FullName
            Add-UniqueValue -List $paths -Value $relative
        }

    return @($paths)
}

function Get-AutoPackageConfig {
    param([Parameter(Mandatory = $true)][string]$Root)

    $includePaths = [System.Collections.Generic.List[string]]::new()
    $softwareName = Split-Path -Leaf $Root
    $version = "0.0.0"

    $pyprojectPath = Join-Path $Root "pyproject.toml"
    if (Test-Path -LiteralPath $pyprojectPath) {
        $pyprojectText = Get-Content -LiteralPath $pyprojectPath -Raw
        $projectSection = Get-TomlSection -Text $pyprojectText -Name "project"
        $pyName = Get-TomlString -Section $projectSection -Key "name"
        $pyVersion = Get-TomlString -Section $projectSection -Key "version"
        if (-not [string]::IsNullOrWhiteSpace($pyName)) {
            $softwareName = $pyName
        }
        if (-not [string]::IsNullOrWhiteSpace($pyVersion)) {
            $version = $pyVersion
        }

        foreach ($path in (Get-PythonPackagePaths -Root $Root -PyProjectText $pyprojectText)) {
            Add-UniqueValue -List $includePaths -Value $path
        }

        Add-ExistingPaths -List $includePaths -Root $Root -RelativePaths @(
            "pyproject.toml",
            "setup.cfg",
            "setup.py",
            "requirements.txt",
            "requirements-dev.txt",
            "pytest.ini",
            "tox.ini",
            "README.md",
            "README.rst",
            "LICENSE",
            "LICENSE.txt"
        )
    }

    $packageJsonPath = Join-Path $Root "package.json"
    if (Test-Path -LiteralPath $packageJsonPath) {
        $packageJson = Get-Content -LiteralPath $packageJsonPath -Raw | ConvertFrom-Json
        $nodeName = Get-ConfigValue -Config $packageJson -Name "name" -Default $null
        $nodeVersion = Get-ConfigValue -Config $packageJson -Name "version" -Default $null
        if (-not [string]::IsNullOrWhiteSpace($nodeName)) {
            $softwareName = $nodeName
        }
        if (-not [string]::IsNullOrWhiteSpace($nodeVersion)) {
            $version = $nodeVersion
        }

        Add-ExistingPaths -List $includePaths -Root $Root -RelativePaths @(
            "src",
            "app",
            "pages",
            "components",
            "public",
            "assets",
            "styles",
            "lib",
            "server",
            "client",
            "index.html",
            "package.json",
            "package-lock.json",
            "npm-shrinkwrap.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "tsconfig.json",
            "jsconfig.json",
            "vite.config.js",
            "vite.config.ts",
            "next.config.js",
            "next.config.mjs",
            "next.config.ts",
            "svelte.config.js",
            "astro.config.mjs",
            "README.md",
            "LICENSE"
        )
    }

    if ($includePaths.Count -eq 0) {
        Add-ExistingPaths -List $includePaths -Root $Root -RelativePaths @(
            "src",
            "app",
            "lib",
            "bin",
            "cmd",
            "pkg",
            "internal",
            "tests",
            "README.md",
            "LICENSE"
        )
    }

    return [pscustomobject]@{
        softwareName = $softwareName
        version = $version
        outputFolder = "software_packages"
        includePaths = @($includePaths)
        initialInputPaths = @(Get-AutoInitialInputPaths -Root $Root)
    }
}

function Resolve-PathListSetting {
    param(
        $ConfiguredValue,
        [string[]]$AutoPaths,
        $AdditionalValue
    )

    $resolved = [System.Collections.Generic.List[string]]::new()
    $configuredPaths = @(Convert-ToStringArray $ConfiguredValue)

    if ($configuredPaths.Count -eq 0) {
        foreach ($path in $AutoPaths) {
            Add-UniqueValue -List $resolved -Value $path
        }
    } else {
        foreach ($path in $configuredPaths) {
            if (Test-AutoValue $path) {
                foreach ($autoPath in $AutoPaths) {
                    Add-UniqueValue -List $resolved -Value $autoPath
                }
            } else {
                Add-UniqueValue -List $resolved -Value $path
            }
        }
    }

    foreach ($path in @(Convert-ToStringArray $AdditionalValue)) {
        Add-UniqueValue -List $resolved -Value $path
    }

    return @($resolved)
}

function Get-MergedPackageConfig {
    param(
        [Parameter(Mandatory = $true)]$RawConfig,
        [Parameter(Mandatory = $true)]$AutoConfig,
        [Parameter(Mandatory = $true)][bool]$AutoEnabled
    )

    $autoName = if ($AutoEnabled) { $AutoConfig.softwareName } else { "software" }
    $autoVersion = if ($AutoEnabled) { $AutoConfig.version } else { "0.0.0" }
    $autoIncludePaths = if ($AutoEnabled) { @($AutoConfig.includePaths) } else { @() }
    $autoInitialInputs = if ($AutoEnabled) { @($AutoConfig.initialInputPaths) } else { @() }

    $nameValue = Get-ConfigValue -Config $RawConfig -Name "softwareName" -Default $null
    $versionValue = Get-ConfigValue -Config $RawConfig -Name "version" -Default $null
    $outputValue = Get-ConfigValue -Config $RawConfig -Name "outputFolder" -Default $null

    $softwareName = if ((Test-AutoValue $nameValue) -and $AutoEnabled) { $autoName } elseif (Test-AutoValue $nameValue) { "software" } else { [string]$nameValue }
    $version = if ((Test-AutoValue $versionValue) -and $AutoEnabled) { $autoVersion } elseif (Test-AutoValue $versionValue) { "0.0.0" } else { [string]$versionValue }
    $outputFolder = if (Test-AutoValue $outputValue) { $AutoConfig.outputFolder } else { [string]$outputValue }

    $includePaths = Resolve-PathListSetting `
        -ConfiguredValue (Get-ConfigValue -Config $RawConfig -Name "includePaths" -Default $null) `
        -AutoPaths $autoIncludePaths `
        -AdditionalValue (Get-ConfigValue -Config $RawConfig -Name "additionalIncludePaths" -Default $null)

    $initialInputPaths = Resolve-PathListSetting `
        -ConfiguredValue (Get-ConfigValue -Config $RawConfig -Name "initialInputPaths" -Default $null) `
        -AutoPaths $autoInitialInputs `
        -AdditionalValue (Get-ConfigValue -Config $RawConfig -Name "additionalInitialInputPaths" -Default $null)

    $defaultExcludeNames = @(
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        "htmlcov",
        "coverage",
        "site",
        "software_packages"
    )

    $defaultExcludeFileExtensions = @(".pyc", ".pyo", ".log", ".tmp", ".temp")

    $defaultExcludePathGlobs = @(
        "_*",
        "run_*",
        "*_run_*",
        "probe_out_*",
        "output_*",
        "*_outputs",
        "output",
        "outputs",
        "results",
        "tmp",
        "temp",
        "*.egg-info",
        "package_app.ps1",
        "package_app.bat",
        "package_app.config.json"
    )

    $excludeNames = [System.Collections.Generic.List[string]]::new()
    foreach ($name in $defaultExcludeNames + (Convert-ToStringArray (Get-ConfigValue -Config $RawConfig -Name "excludeNames" -Default $null))) {
        Add-UniqueValue -List $excludeNames -Value $name
    }
    Add-UniqueValue -List $excludeNames -Value (Split-Path -Leaf $outputFolder)

    $excludeFileExtensions = [System.Collections.Generic.List[string]]::new()
    foreach ($extension in $defaultExcludeFileExtensions + (Convert-ToStringArray (Get-ConfigValue -Config $RawConfig -Name "excludeFileExtensions" -Default $null))) {
        Add-UniqueValue -List $excludeFileExtensions -Value $extension
    }

    $excludePathGlobs = [System.Collections.Generic.List[string]]::new()
    foreach ($glob in $defaultExcludePathGlobs + (Convert-ToStringArray (Get-ConfigValue -Config $RawConfig -Name "excludePathGlobs" -Default $null))) {
        Add-UniqueValue -List $excludePathGlobs -Value $glob
    }

    return [pscustomobject]@{
        softwareName = $softwareName
        version = $version
        outputFolder = $outputFolder
        includePaths = @($includePaths)
        initialInputPaths = @($initialInputPaths)
        excludeNames = @($excludeNames)
        excludeFileExtensions = @($excludeFileExtensions)
        excludePathGlobs = @($excludePathGlobs)
    }
}

function Test-GlobMatch {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [string[]]$Globs
    )

    $normalized = (Convert-ToZipPath $RelativePath).Trim("/")
    foreach ($glob in $Globs) {
        $pattern = (Convert-ToZipPath $glob).Trim("/")
        if ($normalized -like $pattern -or $normalized -like "$pattern/*") {
            return $true
        }
    }
    return $false
}

function Test-Excluded {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileSystemInfo]$Item,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)]$Config
    )

    $parts = (Convert-ToZipPath $RelativePath).Split("/", [System.StringSplitOptions]::RemoveEmptyEntries)
    foreach ($part in $parts) {
        if ($part.StartsWith(".")) {
            return $true
        }
        foreach ($name in @($Config.excludeNames)) {
            if ($part -ieq $name) {
                return $true
            }
        }
    }

    if (Test-GlobMatch -RelativePath $RelativePath -Globs @($Config.excludePathGlobs)) {
        return $true
    }

    if (-not $Item.PSIsContainer) {
        foreach ($extension in @($Config.excludeFileExtensions)) {
            if ($Item.Extension -ieq $extension) {
                return $true
            }
        }
    }

    return $false
}

function Copy-PackagePath {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$StageRoot,
        [Parameter(Mandatory = $true)]$Config,
        [System.Collections.Generic.List[string]]$Included
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Warning "Configured path does not exist and will be skipped: $Source"
        return
    }

    $sourceItem = Get-Item -LiteralPath $Source -Force
    $items = if ($sourceItem.PSIsContainer) {
        Get-ChildItem -LiteralPath $Source -Force -Recurse
    } else {
        @($sourceItem)
    }

    foreach ($item in $items) {
        $relative = Convert-ToRelativePath -Root $Root -Path $item.FullName
        if (Test-Excluded -Item $item -RelativePath $relative -Config $Config) {
            continue
        }

        $destination = Join-Path $StageRoot $relative
        if ($item.PSIsContainer) {
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
            continue
        }

        $destinationDir = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
        Copy-Item -LiteralPath $item.FullName -Destination $destination -Force
        Add-UniqueValue -List $Included -Value $relative
    }
}

$startRoot = if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    Find-RepositoryRoot -StartPath (Get-Location).Path
} else {
    Find-RepositoryRoot -StartPath $ProjectRoot
}
$root = Get-FullPath $startRoot

$configFullPath = if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $candidateConfig = Join-Path $root "package_app.config.json"
    if (Test-Path -LiteralPath $candidateConfig) { $candidateConfig } else { $null }
} else {
    if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
        Get-FullPath $ConfigPath
    } else {
        Get-FullPath (Join-Path $root $ConfigPath)
    }
}

$rawConfig = [pscustomobject]@{}
if ($null -ne $configFullPath) {
    if (-not (Test-Path -LiteralPath $configFullPath)) {
        throw "Config file not found: $configFullPath"
    }
    $rawConfig = Get-Content -LiteralPath $configFullPath -Raw | ConvertFrom-Json
}

$autoDetectValue = Get-ConfigValue -Config $rawConfig -Name "autoDetect" -Default $true
$autoEnabled = -not $NoAutoDetect -and ([string]$autoDetectValue).ToLowerInvariant() -notin @("false", "0", "no")
$autoConfig = if ($autoEnabled) {
    Get-AutoPackageConfig -Root $root
} else {
    [pscustomobject]@{
        softwareName = "software"
        version = "0.0.0"
        outputFolder = "software_packages"
        includePaths = @()
        initialInputPaths = @()
    }
}

$config = Get-MergedPackageConfig -RawConfig $rawConfig -AutoConfig $autoConfig -AutoEnabled $autoEnabled
if (@($config.includePaths).Count -eq 0) {
    throw "No app paths were detected. Add includePaths to package_app.config.json or run with -ProjectRoot pointing at the app root."
}

$softwareName = [string]$config.softwareName
$version = [string]$config.version
$safeName = ($softwareName -replace '[^\w.-]+', '_').Trim("_")
$safeVersion = ($version -replace '[^\w.-]+', '_').Trim("_")
if ([string]::IsNullOrWhiteSpace($safeName)) {
    $safeName = "software"
}
if ([string]::IsNullOrWhiteSpace($safeVersion)) {
    $safeVersion = "0.0.0"
}

$zipName = "{0}_{1}_{2}.zip" -f (Get-Date -Format "yyMMdd"), $safeName, $safeVersion
$outputDir = Resolve-UnderRoot -Root $root -Path ([string]$config.outputFolder)

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$zipPath = Join-Path $outputDir $zipName
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("package_app_{0}_{1}" -f $safeName, [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
$included = [System.Collections.Generic.List[string]]::new()

try {
    foreach ($path in @($config.includePaths)) {
        $source = Resolve-UnderRoot -Root $root -Path ([string]$path)
        Copy-PackagePath -Source $source -Root $root -StageRoot $stageRoot -Config $config -Included $included
    }

    foreach ($path in @($config.initialInputPaths)) {
        $source = Resolve-UnderRoot -Root $root -Path ([string]$path)
        Copy-PackagePath -Source $source -Root $root -StageRoot $stageRoot -Config $config -Included $included
    }

    if ($included.Count -eq 0) {
        throw "No files matched the packaging configuration."
    }

    if ($DryRun) {
        Write-Host "Project root: $root"
        Write-Host "Detected software: $softwareName $version"
        Write-Host "Dry run only. Would create: $zipPath"
        $included | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
        exit 0
    }

    Compress-Archive -Path (Join-Path $stageRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host "Project root: $root"
    Write-Host "Detected software: $softwareName $version"
    Write-Host "Created package: $zipPath"
    Write-Host ("Included files: {0}" -f ($included | Sort-Object -Unique).Count)
} finally {
    if (Test-Path -LiteralPath $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}
