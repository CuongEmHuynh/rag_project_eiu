SELECT TOP 15 D.Id, KeyFileId, [Page], D.[No], Author, Summary, D.Column1 as 'DateDocument',RecordId,
    R.Code as 'RecordCode', R.Name as 'RecordName', -- ho so 
    RT.Code as 'RecordTypeCode', RT.Name as 'RecordTypeName', -- loai ho so
    B.Code as 'BoxCode',  -- hop ho so
    T.Code as 'TableOfContentCode', T.Name as 'TableOfContentName',  -- muc luc
    DB.Code as 'DocumentBlockCode', DB.Name as 'DocumentBlockName',  -- du an hang muc
    F.Id as 'FileIdMinio', CONCAT(CAST(F.Id AS varchar(36)), '_', ISNULL(F.Name, '')) AS 'FileNameMinio', 'host'+F.[Path] as 'FilePathMinio'
    From Documents D

    LEFT JOIN FileManagementBlobFiles F ON D.KeyFileId=F.MasterKeyId
    LEFT JOIN Records R ON D.RecordId=R.Id
    LEFT JOIN RecordsTypes RT ON R.RecordTypeId=RT.Id
    LEFT JOIN Boxes B ON R.BoxId=B.Id
    LEFT JOIN TableOfContents T ON R.TableOfContentId=T.Id
    LEFT JOIN DocumentBlocks DB ON T.DocumentBlockId=DB.Id
    LEFT JOIN Organizations O ON DB.OrganizationId=O.Id
    where   D.Id in 
('19535402-9EE0-7914-5984-3A18B9689C72',
'2E29EC6E-4E1A-D430-B0AA-3A18B9689C74',
'B5A5D4B5-788A-3612-BF41-3A18B9689C74',
'1FDCDD5B-7DAD-E024-5E60-3A18B9689C75',
'DDFBDF14-D24F-082F-60D5-3A18B9689C75',
'2D487D1D-987E-428E-8B7D-3A18B9689C75',
'D5211974-AF44-001F-C60D-3A18B9689C76',
'EF0359F7-E2A8-B9FC-4844-3A18B9689C78',
'0CD43711-1C4E-ED47-267C-3A18B9689C79',
'ABFE6B47-5F61-DF57-BA65-3A18B9689C79',
'7AD2A9E5-E00B-33A6-9C14-3A18B9689C7A',
'B684632F-E7E8-AE1F-A874-3A18B9689C7A',
'5CE1C745-8318-A21F-EC0D-3A18B9689C7A',
'1CFB9318-6ED2-D991-1A21-3A18B9689C7B',
'0103F637-38D7-8E0B-262B-3A18B9689C7B')
