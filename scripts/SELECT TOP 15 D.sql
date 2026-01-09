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
    where   D.Id='2e29ec6e-4e1a-d430-b0aa-3a18b9689c74'

    --- Id Test--
    -- DocumentID:2e29ec6e-4e1a-d430-b0aa-3a18b9689c74