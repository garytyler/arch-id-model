$src = "..\input\architectural-styles-dataset"
Get-ChildItem -Path $src -Filter *.jpg -Recurse | where {$_.extension -eq ".JPG"} | % {
    $guid = New-Guid
    $base = $guid -replace '-',''
    $name = "$base.jpg"
    Rename-Item -Path $_.FullName -NewName $name
}
